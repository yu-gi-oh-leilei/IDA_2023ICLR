import math
import torch
import numpy as np
import random
import torch.nn.functional as F
import torchvision.utils


class CMix(object):  # custom mix
    def __init__(self, mode='cmix', grids=(4,), n_grids=(0,), mix_prob=1., is_pair=False):
        # For a c-mixed sample, $grids samples are mixed. If $grids is (4, 9), 4 samples and 9 samples
        # from the same batch will be mixed randomly to generate a sample respective.
        # $n_grids denotes the number of c-mixed sample used in this batch. If $grids is (4, 9) and $n_grids is (4, 2),
        # 4 and 2 c-mixed samples will be generated where the former 4 samples are mixed by 4 random samples and
        # the latter by 9 random samples.
        super(CMix, self).__init__()

        mode = self.checkMode(mode)

        Mode = {
            'cmix': self.cmix,
            'cmix_ab1': self.cmix_ab1,
            'cmix_mixup': self.cmix_mixup
            }

        self.mix_prob = mix_prob
        self.grids = grids
        self.n_grids = n_grids
        self.mix = Mode[mode]
        self.is_pair = is_pair

    def cmix(self, input, target, ):
        if np.random.rand(1) > self.mix_prob:
            return input, target, {}

        bs = input.shape[0]
        rand_ind = random.sample(range(bs), bs)
        input_a, input_b, target_a, target_b = handle_pair(input, target, self.is_pair)
        flag = torch.ones((bs, 1), device=input.device)
        for g, ng in zip(self.grids, self.n_grids):
            # g_h = g_w = math.sqrt(g)
            # assert g_h.is_integer(), "$grids's sqrt is excepted to be an integer."
            ng_ref = bs // g
            if ng == 0:
                if len(self.grids) == 1:
                    ng = ng_ref
                else:
                    print('argument error, do not execute c-mix')
                    break
            # assert ng_ref >= ng, "n_grids must less than or equal to batch_size//grids"
            if ng > ng_ref:
                rand_ind = np.asarray([random.sample(range(bs), bs) for i in range(ng)]).reshape(-1)
            rand_ind = rand_ind[:ng*g]
            input_mix_g, target_mix_g, _ = self.mix_fn(input_b[rand_ind], target_b[rand_ind], grid=g, n_grid=ng)


            # input_a = torch.cat([input_a, input_mix_g], dim=0)
            ####################
            ## cutout using
            # NH = [1]
            # S = (.1, .2, .3, .4, .5)
            # nh = NH[random.sample(range(len(NH)), 1)[0]]
            # co = Cutout(n_holes=nh, scales=S)
            # input_a = co(input_a.view(-1, input_a.shape[-2], input_a.shape[-1])).contiguous()
            # input_a = input_a.view(bs, 3, input_a.shape[-2], input_a.shape[-1])
            input_a = torch.cat([input_a, input_mix_g], dim=0)

            target_a = torch.cat([target_a, target_mix_g], dim=0)
            flag = torch.cat([flag, torch.ones((ng, 1), device=flag.device)*g], dim=0)
        tmp = {'flag': flag, 'rand_ind': rand_ind}
        return input_a, target_a, tmp

    def mix_fn(self, input, target, grid, n_grid):
        bs, c,  h, w = input.shape
        assert math.sqrt(grid).is_integer(), "$grids's sqrt is excepted to be an integer."
        g_h = g_w = int(math.sqrt(grid))
        input = F.interpolate(input, (h//g_h, w//g_w), mode='bilinear', align_corners=True)  # g*ng, C, h', w'
        input_mix = torchvision.utils.make_grid(input, nrow=int(g_w), padding=0)  # C, ng*h, w
        input_mix = input_mix.split(h, dim=1)  # tuple: ng, (C, h, w)
        input_mix = torch.stack(input_mix, dim=0)
        target_mix = target.view(n_grid, grid, -1).sum(1)
        target_mix[target_mix>0] = 1

        return input_mix, target_mix, {}

    def cmix_ab1(self, input, target, rand_bbox_flag=False, rand_scale=False, num=1):
        if hasattr(self, 'rand_bbox_flag'):
            rand_bbox_flag = self.rand_bbox_flag
        if hasattr(self, 'rand_scale'):
            rand_scale = self.rand_scale
        if hasattr(self, 'num'):
            num = self.num


        bs, c, h, w = input.shape
        grid = self.grids[0]
        n_grid = self.n_grids[0]
        ## todo: cutout-style
        if rand_bbox_flag:  # if Ture, num===1
            lam = random.uniform(.5, 1) if rand_scale else .5
            bbx1, bby1, bbx2, bby2 = rand_bbox((h, w), lam)
            tmp = torch.zeros(n_grid, c, h, w, device=input.device)
            rand_ind = random.sample(range(bs), n_grid)
            h2, w2 = bbx2-bbx1, bby2-bby1
            input_small = F.interpolate(input, (h2, w2), mode='bilinear', align_corners=True)
            tmp[:, :, bbx1:bbx2, bby1:bby2] = input_small[rand_ind]
            target_mix = target[rand_ind]

            input_mix = torch.cat((input, tmp), dim=0)
            target_mix = torch.cat((target, target_mix), dim=0)
            tmp = {'rand_ind': rand_ind}
        else:
            if self.num == -5:
                num = random.sample(range(4), 1)[0] + 1
            assert math.sqrt(grid).is_integer(), "$grids's sqrt is excepted to be an integer."
            g_h = g_w = int(math.sqrt(grid))
            h2, w2 = h // g_h, w // g_w
            au_ = n_grid*g_h*g_w
            tmp = torch.zeros(au_, c, h2, w2, device=input.device)

            # rand_ind = [[random.randrange(0, 4*n_grid) for i in range(n_grid)] for i in range(num)]  # au
            rand_ind = [random.sample(range(i*4, (i+1)*4), grid) for i in range(n_grid)]  # au
            rand_ind = torch.tensor(rand_ind, device=input.device)[:, :num].reshape(-1)
            # scale_rand = torch.arange(n_grid, device=input.device) * grid
            # rand_ind = scale_rand + rand_ind
            rand_ind2 = random.sample(range(bs), n_grid*num)
            input_small = F.interpolate(input, (h2, w2), mode='bilinear', align_corners=True)
            tmp[rand_ind] = input_small[rand_ind2]
            input_mix = torchvision.utils.make_grid(tmp, nrow=int(g_w), padding=0)  # C, ng*h, w
            input_mix = input_mix.split(h, dim=1)  # tuple: ng, (C, h, w)
            input_mix = torch.stack(input_mix, dim=0)

            target_mix = target[rand_ind2].view(n_grid, num, -1).sum(1)
            target_mix[target_mix > 0] = 1

            input_mix = torch.cat((input, input_mix), dim=0)
            target_mix = torch.cat((target, target_mix), dim=0)

            tmp = {'rand_ind': rand_ind2}
        return input_mix, target_mix, tmp

    def cmix_mixup(self, input, target, alpha=.1):
        if np.random.rand(1) > self.mix_prob:
            return input, target, {}

        bs = input.shape[0]
        rand_ind = random.sample(range(bs), bs)
        input_a, input_b, target_a, target_b = handle_pair(input, target, self.is_pair)

        flag = torch.ones((bs, 1), device=input.device)
        for g, ng in zip(self.grids, self.n_grids):
            # g_h = g_w = math.sqrt(g)
            # assert g_h.is_integer(), "$grids's sqrt is excepted to be an integer."
            ng_ref = bs // g
            if ng == 0:
                if len(self.grids) == 1:
                    ng = ng_ref
                else:
                    print('argument error, do not execute c-mix')
                    break
            # assert ng_ref >= ng, "n_grids must less than or equal to batch_size//grids"
            if ng > ng_ref:
                rand_ind = np.asarray([random.sample(range(bs), bs) for i in range(ng)]).reshape(-1)
            rand_ind = rand_ind[:ng*g]

            ## mixup
            input_b = input_b[rand_ind]
            target_b = target_b[rand_ind]
            lam = np.random.beta(alpha, alpha)
            # rand_ind2 = random.sample(range(input_b.shape[0]), input_b.shape[0])
            rand_ind3 = [random.sample(range(i * g, (i + 1) * g), g) for i in range(ng)]
            rand_ind3 = np.asarray(rand_ind3).reshape((-1,))

            input_b = input_b * lam + input_b[rand_ind3] * (1 - lam)

            input_mix_g, target_mix_g, _ = self.mix_fn(input_b, target_b, grid=g, n_grid=ng)
            input_a = torch.cat([input_a, input_mix_g], dim=0)
            target_a = torch.cat([target_a, target_mix_g], dim=0)
            flag = torch.cat([flag, torch.ones((ng, 1), device=flag.device)*g], dim=0)
        tmp = {'flag': flag, 'rand_ind': rand_ind}
        return input_a, target_a, tmp

    def checkMode(self, mode):
        if '--' in mode:  # like cmix_ab1--num=2--rand_scale=True
            str_list = mode.split('--')
            mode = str_list[0]
            for s in str_list[1:]:
                # varb, val = s.split('=')
                # exec(f"self.{varb} = {val}")
                exec(f"self.{s}")
        return mode

class Mix(object):
    def __init__(self, mode='', alpha=1., mix_prob=1., is_pair=False):
        super(Mix, self).__init__()

        assert alpha > 0, "alpha in beta distribution should be larger than zero."

        Mode = {
            '': self.no_mix,
            'mixup': self.mixup,
            'cutmix': self.cutmix,
            'mixup_ml': self.mixup_ml,
            }

        self.is_pair = is_pair > 0
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.mix = Mode[mode]

    def no_mix(self, input, target, ):
        tmp = {}
        return input, target, tmp

    def mixup(self, input, target, ):
    # from https://github.com/facebookresearch/mixup-cifar10/blob/eaff31ab397a90fbc0a4aac71fb5311144b3608b/train.py
        if np.random.rand(1) > self.mix_prob:
            return input, target, {}
        bs = input.shape[0]
        rand_index = random.sample(range(bs), bs)
        lam = np.random.beta(self.alpha, self.alpha)

        input_a, input_b, target_a, target_b = handle_pair(input, target, self.is_pair)

        input_mixed = input_a*lam + input_b[rand_index]*(1-lam)
        target_mixed = target_a*lam + target_b[rand_index]*(1-lam)

        tmp = {'lam': lam}
        return input_mixed, target_mixed, tmp

    def cutmix(self, input, target, ):
    # from https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py
        if np.random.rand(1) > self.mix_prob:
            return input, target, {}
        bs, _, h, w = input.shape
        rand_index = random.sample(range(bs), bs)
        lam = np.random.beta(self.alpha, self.alpha)

        input_a, input_b, target_a, target_b = handle_pair(input, target, self.is_pair)

        bbx1, bby1, bbx2, bby2 = rand_bbox((h, w), lam)
        input_a[:, :, bbx1:bbx2, bby1:bby2] = input_b[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        target_mixed = target_a*lam + target_b[rand_index]*(1-lam)

        tmp = {'lam': lam}
        return input_a, target_mixed, tmp

    def mixup_ml(self, input, target, ):
    # from https://github.com/facebookresearch/mixup-cifar10/blob/eaff31ab397a90fbc0a4aac71fb5311144b3608b/train.py
        if np.random.rand(1) > self.mix_prob:
            return input, target, {}
        bs = input.shape[0]
        rand_index = random.sample(range(bs), bs)
        lam = np.random.beta(self.alpha, self.alpha)

        input_a, input_b, target_a, target_b = handle_pair(input, target, self.is_pair)

        input_mixed = input_a*lam + input_b[rand_index]*(1-lam)
        target_mixed = target_a + target_b[rand_index]
        target_mixed[target_mixed> 1] = 0

        tmp = {'lam': lam}
        return input_mixed, target_mixed, tmp

def handle_pair(input, target, is_pair):
    # is_pair == False: mixing from the same batch
    # is_pair == True: mixing example randomly sampled from the entire data set
    if not is_pair:
        input_a = input
        input_b = input.clone()
        target_a = target
        target_b = target.clone()
    else:
        assert input.shape[1] == 2 and target.shape[1] == 2  # input: bs, 2, C, h, w; target: bs, 2, nc
        input_a = input[:, 0]  # bs, C, h, w
        input_b = input[:, 1]  # bs, C, h, w
        target_a = target[:, 0]
        target_b = target[:, 1]
    return input_a, input_b, target_a, target_b

def rand_bbox(size, lam):
    # W = size[2]
    # H = size[3]
    H, W = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        scales (Tuple): The cropped rate is selected from scales.
    """
    def __init__(self, n_holes, scales):
        self.n_holes = n_holes
        self.scales = scales

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.shape[1]
        w = img.shape[2]

        length = random.sample(self.scales, 1)[0] * h

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[int(y1): int(y2), int(x1): int(x2)] = 0.

        mask = torch.from_numpy(mask).to(img.device)
        mask = mask.expand_as(img)
        img = img * mask

        return img
