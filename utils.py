import numpy as np
import concurrent.futures as fs
from numba import jit
import time, sys, os
import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def grab_patches_2(images, targets, n_patches, patch_size, seed=0):
    np.random.seed(seed)
    images = images.transpose(0, 3, 1, 2)

    n_patches_per_row = images.shape[2] - patch_size + 1
    n_patches_per_col = images.shape[3] - patch_size + 1
    n_patches_per_image = n_patches_per_row * n_patches_per_col
    n_patches_total = images.shape[0] * n_patches_per_image
    patch_ids = np.random.choice(n_patches_total, size=n_patches, replace=False)

    patches = np.zeros((n_patches, 3, patch_size, patch_size), dtype=images.dtype)
    patch_classes = np.zeros(n_patches, dtype='int')

    for i_patch, patch_id in enumerate(patch_ids):
        img_id = patch_id % images.shape[0]
        x_id = patch_id // images.shape[0] % n_patches_per_row
        y_id = patch_id // (images.shape[0] * n_patches_per_row)
        patches[i_patch] = images[img_id, :, x_id:x_id+patch_size, y_id:y_id+patch_size]
        patch_classes[i_patch] = targets[img_id]

    return patches, patch_classes

def grab_patches_from_loader(loader, n_patches, image_size, patch_size, n_images=1281159, seed=0):
    n_patches_per_row = image_size - patch_size + 1
    n_patches_per_col = image_size - patch_size + 1
    n_patches_per_image = n_patches_per_row * n_patches_per_col
    n_patches_total = n_images * n_patches_per_image
    patch_ids = np.random.choice(n_patches_total, size=n_patches, replace=False)
    patch_img_ids = patch_ids % n_images
    patch_x_ids = patch_ids // n_images % n_patches_per_row
    patch_y_ids = patch_ids // (n_images * n_patches_per_row)

    patch_batch_ids = patch_img_ids // loader.batch_size
    patch_img_ids_in_batch = patch_img_ids % loader.batch_size

    patches = np.zeros((n_patches, 3, patch_size, patch_size), dtype='float32')
    patch_classes = np.zeros(n_patches, dtype='int')

    patch_count = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cpu().numpy(), targets.cpu().numpy()
        ids = np.argwhere(patch_batch_ids == batch_idx).reshape(-1)
        for id_ in ids:

            patch = inputs[
                patch_img_ids_in_batch[id_],
                :,
                patch_x_ids[id_]:patch_x_ids[id_]+patch_size,
                patch_y_ids[id_]:patch_y_ids[id_]+patch_size
            ]
            patches[patch_count] = patch
            patch_classes[patch_count] = targets[patch_img_ids_in_batch[id_]]
            patch_count += 1
    print(patch_count)

    return patches, patch_classes

def compute_patch_mean_cov(images, patch_size, batch_size=16384):
    images = images.astype('float32')

    mean, cov = None, None
    with torch.no_grad():
        for i_batch in range(np.ceil(images.shape[0] / batch_size).astype('int')):
            images_batch = torch.from_numpy(images[i_batch*batch_size:min((i_batch+1)*batch_size, images.shape[0])]).to(device)
            patches_batch = torch.nn.functional.unfold(images_batch, patch_size, padding=0, stride=1)
            batch_mean = patches_batch.mean(dim=(0, 2)).double()
            mean = batch_mean if mean is None else (i_batch * mean) / (i_batch + 1) + batch_mean / (i_batch + 1)

        dim = 3*patch_size**2
        mean = mean.view(1, dim, 1).contiguous()

        for i_batch in range(np.ceil(images.shape[0] / batch_size).astype('int')):
            images_batch = torch.from_numpy(images[i_batch*batch_size:min((i_batch+1)*batch_size, images.shape[0])]).to(device)
            patches_batch = torch.nn.functional.unfold(images_batch, patch_size, padding=0, stride=1).double()
            patches_batch -=  mean
            patches_batch = patches_batch.transpose(0, 1).contiguous().view(dim, -1)
            batch_cov = torch.mm(patches_batch, patches_batch.t()) / patches_batch.size(1)
            cov = batch_cov if cov is None else (i_batch * cov) / (i_batch + 1) + batch_cov / (i_batch + 1)

        mean = mean.view(-1).contiguous()

    return mean.cpu().numpy(), cov.cpu().numpy()

def whiten_and_normalize_patches(patches, mean, cov, min_divisor=1e-8, zca_bias=0.001):
    if (patches.dtype == 'uint8'):
        patches = patches.astype('float64')
        patches /= 255.0
    patches = patches.astype('float64')
    print("zca bias", zca_bias)

    orig_shape = patches.shape
    patches = patches.reshape(patches.shape[0], -1)

    patches = patches - mean[np.newaxis, :]

    (E,V) = np.linalg.eig(cov)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E)
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
    patches_whitened = (patches).dot(global_ZCA).dot(global_ZCA.T)

    patches_whitened_normalized = patches_whitened / (np.linalg.norm(patches_whitened, axis=1, keepdim=True) + min_divisor)

    return patches_whitened_normalized.reshape(orig_shape).astype('float32')



def normalize_patches(patches, min_divisor=1e-8, zca_bias=0.001, mean_rgb=np.array([0,0,0]), zca_whitening=True):
    if (patches.dtype == 'uint8'):
        patches = patches.astype('float64')
        patches /= 255.0
    print("zca bias", zca_bias)
    n_patches = patches.shape[0]
    orig_shape = patches.shape
    patches = patches.reshape(patches.shape[0], -1)

    # Zero mean every feature
    patches = patches - np.mean(patches, axis=1)[:,np.newaxis]

    # Added by Louis : Statistical zero mean for ZCA
    patches = patches - np.mean(patches, axis=0, keepdims=True)

    # Normalize
    patch_norms = np.linalg.norm(patches, axis=1)

    # Get rid of really small norms
    #patch_norms[np.where(patch_norms < min_divisor)] = 1

    # Make features unit norm
    #patches = patches/patch_norms[:,np.newaxis]

    if zca_whitening:
        patchesCovMat = 1.0/n_patches * patches.T.dot(patches)

        (E,V) = np.linalg.eig(patchesCovMat)

        E += zca_bias
        sqrt_zca_eigs = np.sqrt(E)
        inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
        global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
        patches_normalized = (patches).dot(global_ZCA).dot(global_ZCA.T)
    else:
        patches_normalized = patches

    # Normalize
    patch_normalized_norms = np.linalg.norm(patches_normalized, axis=1) #EO

    # Get rid of really small norms
    patch_normalized_norms[np.where(patch_normalized_norms < min_divisor)] = 1 #EO
    patches_normalized = patches_normalized / patch_normalized_norms[:, np.newaxis]# EO

    return patches_normalized.reshape(orig_shape).astype('float32')

def chunk_idxs(size, chunks):
    chunk_size  = int(np.ceil(size/chunks))
    idxs = list(range(0, size+1, chunk_size))
    if (idxs[-1] != size):
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))


@jit(nogil=True, cache=True)
def __grab_patches(images, random_idxs, patch_size=6, tot_patches=1e6, seed=0, scale=0):
    patches = np.zeros((len(random_idxs), images.shape[1], patch_size, patch_size), dtype=images.dtype)
    for i, (im_idx, idx_x, idx_y) in enumerate(random_idxs):
        out_patch = patches[i, :, :, :]
        im = images[im_idx]
        grab_patch_from_idx(im, idx_x, idx_y, patch_size, out_patch)
    return patches

def grab_patches(images, patch_size=6, tot_patches=5e5, seed=0, max_threads=50, scale=0, rgb=True):
    if (rgb):
        images = images.transpose(0, 3, 1, 2)
    idxs = chunk_idxs(images.shape[0], max_threads)
    tot_patches = int(tot_patches)
    patches_per_thread = int(tot_patches/max_threads)
    np.random.seed(seed)
    seeds = np.random.choice(int(1e5), len(idxs), replace=False)
    dtype = images.dtype

    tot_patches = int(tot_patches)



    with fs.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for i,(sidx, eidx) in enumerate(idxs):
            images.shape[0]
            im_idxs = np.random.choice(images[sidx:eidx, :].shape[0], patches_per_thread)
            idxs_x = np.random.choice(int(images.shape[2]) - patch_size - 1, tot_patches)
            idxs_y = np.random.choice(int(images.shape[3]) - patch_size - 1, tot_patches)
            idxs_x += int(np.ceil(patch_size/2))
            idxs_y += int(np.ceil(patch_size/2))
            random_idxs =  list(zip(im_idxs, idxs_x, idxs_y))

            futures.append(executor.submit(__grab_patches, images[sidx:eidx, :],
                                           patch_size=patch_size,
                                           random_idxs=random_idxs,
                                           tot_patches=patches_per_thread,
                                           seed=seeds[i],
                                           scale=scale
                                            ))
        results = np.vstack(list(map(lambda x: x.result(), futures)))
    idxs = np.random.choice(results.shape[0], results.shape[0], replace=False)
    return results[idxs], idxs

@jit(nopython=True, nogil=True)
def grab_patch_from_idx(im, idx_x, idx_y, patch_size, outpatch):
    sidx_x = int(idx_x - patch_size/2)
    eidx_x = int(idx_x + patch_size/2)
    sidx_y = int(idx_y - patch_size/2)
    eidx_y = int(idx_y + patch_size/2)
    outpatch[:,:,:] = im[:, sidx_x:eidx_x, sidx_y:eidx_y]
    return outpatch


TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except ValueError:
    term_width = 120

def progress_bar(current, total, msg=None, hide=False):
    if hide:
        return
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()



def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def compute_channel_PCA(data_loader, transformation):
    """Compute PCA along channel dimension."""
    mean, cov = None, None
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            inputs = inputs.cuda()
            inputs = transformation(inputs)
            batch_mean = inputs.mean(dim=(0, 2, 3)).double()
            mean = batch_mean if mean is None else (i * mean) / (i + 1) + batch_mean / (i + 1)

        dim = mean.size(0)
        mean = mean.float().view(1, dim, 1, 1)

        for i, (inputs, _) in enumerate(data_loader):
            inputs = inputs.cuda()
            inputs = transformation(inputs) - mean
            inputs = inputs.transpose(0, 1).contiguous().view(dim, -1).double()
            batch_cov = torch.mm(inputs, inputs.t()) / inputs.size(1)
            cov = batch_cov if cov is None else (i * cov) / (i + 1) + batch_cov / (i + 1)
        cov = cov.float()

        eigenvalues, eigenvectors = torch.symeig(cov, eigenvectors=True)
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1).t()
        mean = mean.view(-1).contiguous()

    return mean, eigenvalues, eigenvectors


def update_batch_norms(batch_norm_couples, i_start, i_end):
    for bn, bagged_bn in batch_norm_couples:
        bagged_bn.running_mean.data[i_start:i_end] = bn.running_mean.data.cpu()
        bagged_bn.running_var.data[i_start:i_end] = bn.running_var.data.cpu()
        bagged_bn.weight.data[i_start:i_end] = bn.weight.data.cpu()
        bagged_bn.bias.data[i_start:i_end] = bn.bias.data.cpu()


def update_classifiers(classifier_couples, i_start, i_end, n_bagged):
    for cfier, bagged_cfier in classifier_couples:
        bagged_cfier.weight.data[:,i_start:i_end] = 1. / n_bagged * cfier.weight.data
        bagged_cfier.bias.data += 1. / n_bagged * cfier.bias.data

def correct_topk(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum().item()
            res.append(correct_k)
    return res


