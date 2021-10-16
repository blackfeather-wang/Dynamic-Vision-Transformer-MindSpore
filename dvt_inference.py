import os
import math
import numpy as np


# def dynamic_evaluate(model, test_loader, val_loader, args, flops):
def dynamic_evaluate(val_pred, val_target, test_pred, test_target, flops):
    tester = Tester()
    # if os.path.exists(os.path.join(args.save, 'logits_single.pth')):
    #     val_pred, val_target, test_pred, test_target = \
    #         torch.load(os.path.join(args.save, 'logits_single.pth'))
    # else:
    #     val_pred, val_target = tester.calc_logit(val_loader)
    #     test_pred, test_target = tester.calc_logit(test_loader)
    #     torch.save((val_pred, val_target, test_pred, test_target),
    #                os.path.join(args.save, 'logits_single.pth'))

    acc_list, exp_flops_list, ratio_list = [], [], []

    samples = {}

    for p in range(1, 40):
        print("*********************")
        _p = p * 1.0 / 20
        probs = [np.exp(np.log(_p) * 1), np.exp(np.log(_p) * 2)]
        probs /= sum(probs)
        # print('probs:', probs)

        acc_val, _, T = tester.dynamic_eval_find_threshold(
            val_pred, val_target, probs, flops)
        # print('acc_val:', acc_val)
        # print('T:', T)
        # print('T.shape:', T.shape)
        # exit(0)
        acc_test, exp_flops, exit_buckets, ratio = tester.dynamic_eval_with_threshold(
            test_pred, test_target, flops, T)
        print('valid acc: {:.3f}, test acc: {:.3f}, test flops: {:.2f}G, ratio:{:.3f}'.format(
            acc_val, acc_test, exp_flops, ratio))

        print('{}\t{}\n'.format(acc_test, exp_flops))

        acc_list.append(acc_test)
        exp_flops_list.append(exp_flops)
        ratio_list.append(ratio)

        samples[p] = exit_buckets

    print('acc:{}\nflops:{}\nratio:{}\n'.format(
        acc_list, exp_flops_list, ratio_list))

    # torch.save([exp_flops_list, acc_list],
    #            os.path.join(args.save, 'dynamic.pth'))
    # torch.save(samples, os.path.join(args.save, 'exit_samples_by_p.pth'))
    return acc_list, exp_flops_list, ratio_list


class Tester(object):
    def __init__(self):
        pass

    def dynamic_eval_find_threshold(self, logits, targets, p, flops):
        """
            logits: m * n * c
            m: Stages
            n: Samples
            c: Classes
        """
        n_stage, n_sample, c = logits.shape
        # print('n_stage:{}, n_sample:{}, c:{}'.format(n_stage, n_sample, c))

        max_preds = logits.max(axis=2, keepdims=False)
        argmax_preds = np.argmax(logits, axis=2)
        # print('max_preds:{}, argmax_preds:{}'.format(max_preds, argmax_preds))
        # print('max_preds.shape:{}, argmax_preds.shape:{}'.format(max_preds.shape, argmax_preds.shape))

        sorted_idx = np.argsort(-max_preds, axis=1)
        # print('sorted_idx:', sorted_idx)

        filtered = np.zeros(n_sample)
        T = np.ones(n_stage) * (1e8)

        for k in range(n_stage - 1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k])
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx]
                        break
            filtered += (max_preds[k] > (T[k]))
            # .type_as(filtered))
        # print('filtered:', filtered)
        # print('filtered.shape:', filtered.shape)
        # print('sum(filtered):', sum(filtered))
        # exit(0)

        T[n_stage - 1] = -1e8  # accept all of the samples at the last stage

        acc_rec, exp = np.zeros(n_stage), np.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i] >= T[k]:  # force the sample to exit at k
                    if int(gold_label) == int(argmax_preds[k][i]):
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0
        for k in range(n_stage):
            _t = 1.0 * exp[k] / n_sample
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, T

    def dynamic_eval_with_threshold(self, logits, targets, flops, T):
        n_stage, n_sample, _ = logits.shape
        # take the max logits as confidence
        max_preds = logits.max(axis=2, keepdims=False)
        argmax_preds = np.argmax(logits, axis=2)

        # for each exit use a bucket to keep track of samples outputing from it
        exit_buckets = {i: {j: []
                            for j in range(n_stage)} for i in range(1000)}

        acc_rec, exp = np.zeros(n_stage), np.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i] >= T[k]:  # force to exit at k
                    _g = int(gold_label)
                    _pred = int(argmax_preds[k][i])
                    if _g == _pred:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    exit_buckets[int(gold_label)][k].append(i)
                    break

        acc_all, sample_all = 0, 0
        for k in range(n_stage):
            _t = exp[k] * 1.0 / n_sample
            sample_all += exp[k]
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, exit_buckets, exp[0]/n_sample


if __name__ == "__main__":
    import torch
    val_pred, val_target, test_pred, test_target = torch.load(os.path.join('/home/cgf/c00500728/code/dvt/dvt_inference/outputs/dvt_t2t_vit_12_inference', 'logits_single.pth'))
    flops1 = 1.145
    flops2 = 4.608
    flops = [flops1, flops1 + flops2]
    val_pred = val_pred.numpy()
    val_target = val_target.numpy()
    test_pred = test_pred.numpy()
    test_target = test_target.numpy()
    print('val_pred.shape:', val_pred.shape)
    print('val_target.shape:', val_target.shape)
    print('test_pred.shape:', test_pred.shape)
    print('test_target.shape:', test_target.shape)
    dynamic_evaluate(val_pred, val_target, test_pred, test_target, flops)
