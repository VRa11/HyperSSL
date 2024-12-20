import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 6
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self,key, run=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            
            
            result =  torch.tensor(self.results)

            best_results = []
            for r in result:
                
                train1 = 100 * r[:, 0].max().item()
                valid = 100 * r[:, 1].max().item()
                train2 = 100 * r[r[:, 1].argmax(), 0].item()
                test = 100 * r[r[:, 3].argmax(), 2].item()
                f1 = r[r[:, 3].argmax(), 3].item()
                precision = r[r[:, 3].argmax(), 4].item()
                recall = r[r[:, 3].argmax(), 5].item()
                best_results.append((train1, valid, train2, test, f1, precision, recall))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            #r = best_result[:, 0]
            #print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            #r = best_result[:, 1]
            #print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            #r = best_result[:, 2]
            #print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final F1: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 5]
            print(f'   Final Precision: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 6]
            print(f'   Final Recall: {r.mean():.2f} ± {r.std():.2f}')
