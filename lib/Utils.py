import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.distributions as dist
def z_score(x, mean, std):
    return (x - mean.reshape((1, 1, -1, 1))) / std.reshape((1, 1, -1, 1))
    #return (x - mean) / std


def z_inverse(x, mean, std):
    return x * std.reshape((1, 1, -1, 1)) + mean.reshape((1, 1, -1, 1))
    #return x * std + mean

class QuantileLoss2(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        preds: tensor of shape (batch, num_horizons, num_sensors, num_quantiles)
        target: tensor of shape (batch, num_horizons, num_sensors, num_quantiles)
        """
        assert not target.requires_grad
        losses = []
        null_val = 0.0
        if np.isnan(null_val):
            mask = ~torch.isnan(target)
        else:
            mask = (target != null_val)
        mask = mask.float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        for i, q in enumerate(self.quantiles):
            errors = target[:,:,:,0] - preds[:,:,:,i]
           
            loss = torch.max((q - 1) * errors, q * errors)
            loss = loss * mask[:,:,:,0]
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
            losses.append(loss)
        losses = sum(losses)
        return losses
    
    
class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        preds: tensor of shape (batch, num_horizons, num_sensors, num_quantiles)
        target: tensor of shape (batch, num_horizons, num_sensors, num_quantiles)
        """
        assert not target.requires_grad
        losses = []
        null_val = 0.0
        if np.isnan(null_val):
            mask = ~torch.isnan(target)
        else:
            mask = (target != null_val)
        mask = mask.float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        for i, q in enumerate(self.quantiles):
            errors = target[:,:,:,0] - preds[:,:,:,i]
           
            loss = torch.max((q - 1) * errors, q * errors)
            loss = loss * mask[:,:,:,0]
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
            losses.append(loss)
        losses = sum(losses)
        return losses.mean()
    
    
class LogUniform(dist.TransformedDistribution):
    def __init__(self, lb, ub):
        super().__init__(dist.Uniform(torch.log(lb), torch.log(ub)), dist.ExpTransform())


class CombinedLoss(nn.Module):
    def __init__(self, *Loss):
        super().__init__()
        self.size = len(Loss)
        self.LossFunc = Loss
        self.weight = torch.zeros(self.size)

    def updateWeight(self, prob_func: dist.TransformedDistribution):
        self.weight = prob_func.sample([self.size])

    def forward(self, preds, target):
        assert not target.requires_grad

        return sum([self.weight[i].item() * self.LossFunc[i](preds, target) for i in range(self.size)])


class MixedLoss(nn.Module):
    def __init__(self, alpha, quantiles):
        super().__init__()
        self.alpha = alpha
        self.quantiles = quantiles

    def forward(self, preds, target):
        quantile_loss = QuantileLoss(self.quantiles)(preds, target)
        return (self.alpha * quantile_loss) + ((1-self.alpha) * masked_mse(preds, target))

    
def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mse(preds, labels, null_val=0.0):
    # null_val=np.nan
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


    
class STDrop(nn.Module):

    def __init__(self, stdrop, sort_rank, s_epoch, cl_epoch, adj, k=30):
        super(STDrop, self).__init__()
        # self.adj = self.get_neighbor()
        self.s_epoch = s_epoch
        self.cl_epoch = cl_epoch
        self.stdrop = stdrop
        self.sort_rank = sort_rank
        self.adj = adj
        self.k = k

    def get_neighbor(self, h=10, w=20):
        adj = torch.zeros((h * w, h * w)).cuda()
        for i in range(h):
            for j in range(w):
                for u in range(h):
                    for v in range(w):
                        adj[i * w + j, u * w + v] = (i - u) ** 2 + (j - v) ** 2
        adj[adj > 3] = 0
        adj[adj != 0] = 1
        return adj

    def get_distance(self, data):
        '''
        data [bz, channel, H, W]
        X [bz, H*W, channel]

        return [bz, H*W, H*W]
        '''
        X = data.detach().reshape(len(data), data.shape[1]*data.shape[2], -1).permute(0, 2, 1)
        X_std = torch.std(X, dim=1, keepdim=True)
        X_mean = torch.mean(X, dim=1, keepdim=True)
        X = (X - X_mean) / (X_std + 1e-6)
        distance = torch.cdist(X, X, p=2)
        return distance

    def get_score(self, distance):
        '''
        distance [bz, H*W, H*W]
        '''
        sort_distance = torch.sort(distance, axis=2)[0]
        # sort_distance [bz, H*W, H*W]
        batch_R = sort_distance.mean(axis=1)[:, self.k]
        # [bz]
        adj_distance = distance * self.adj
        # [bz. H*W, H*W]
        adj_distance[adj_distance == 0] = 1e10

        temp_distance = distance.clone()
        temp_adj_distance = adj_distance.clone()
        self.samples_N = torch.zeros_like(sort_distance.mean(axis=1)).to(sort_distance.device)
        self.neighbor_N = torch.zeros_like(sort_distance.mean(axis=1)).to(sort_distance.device)
        # [bz,H*W]
        for i, x in enumerate(temp_adj_distance):
            x[x < batch_R[i]] = -1
            x[x > 0] = 0
            x[x == -1] = 1
            self.neighbor_N[i] = x.sum(axis=1)
        for i, x in enumerate(temp_distance):
            x[x < batch_R[i]] = -1
            x[x > 0] = 0
            x[x == -1] = 1
            self.samples_N[i] = x.sum(axis=1)
        spatial_score = self.neighbor_N/self.adj.sum(-1)
        temporal_score =  self.samples_N / (self.samples_N + self.samples_N.mean(axis=-1, keepdim=True))
        score = 2 - spatial_score - temporal_score

        return score
    
    def get_quantile_score(self, ypred, ytrue, sort_rank, a_epoch, cl_epoch, qs=[0.1,0.5,0.9]):
        '''
        Quantile loss version 2
        Args:
        ytrue (batch_size, time, node, channel)
        ypred (batch_size, time, node, num_quantiles)
        return out(batch_size, time, node, num_quantiles)
        '''
    #     ytrue[ytrue < 1] = 0
    #     ypred[ypred < 1] = 0    
        assert not ytrue.requires_grad
        losses = []
        null_val = 0.0
        if np.isnan(null_val):
            mask = ~torch.isnan(target)
        else:
            mask = (ytrue != null_val)
        mask = mask.float()
        mask /= torch.mean(mask)    

        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        for i, q in enumerate(qs):
            errors = ytrue[:,:,:,0] - ypred[:,:,:,i]

            loss = torch.max((q - 1) * errors, q * errors)
            loss = loss * mask[:,:,:,0]
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
            losses.append(loss.unsqueeze(-1))
            # 
#         print('torch.cat(losses) shape is ',torch.cat(losses).shape)
        metric_out = torch.cat(losses,dim=3)  # [b,t,n,3]
        
        # sort only temporal   
        spatial_rank_dim = int(metric_out.shape[2] * a_epoch // cl_epoch)
        #print('spatial_rank_dim  is ',spatial_rank_dim)
        temporal_rank_dim = int(metric_out.shape[1] * a_epoch // cl_epoch)
       #$ print('temporal_rank_dim  is ',temporal_rank_dim)
        if sort_rank=='t':
            metric_rank = torch.sort(metric_out, dim=1)[0]
            metric_rank[:,:temporal_rank_dim,:,:] = -1
            metric_rank[:,temporal_rank_dim:,:,:] = 0             
        # sort only spatial
        elif sort_rank=='s':
            metric_out = torch.mean(metric_out, dim=(0,1,3))  # [307]
            # print('metric_out  is ',metric_out)
            #print('torch.sort(metric_out)[0] ',torch.sort(metric_out)[0])
            # print('torch.sort(metric_out)[1] ',torch.sort(metric_out)[1])
            rank =  torch.sort(metric_out)[0]    
            
            #print('rank ',rank)
            boarderline = rank[spatial_rank_dim-1]
            #print('boarderline ',boarderline.item())
            bigger_mask = metric_out>boarderline.item()
            smaller_mask = metric_out<=boarderline.item()
            # print(smaller_mask)
            metric_out[bigger_mask]=0
            metric_out[smaller_mask]=-1 
            # print('metric_out  is ',metric_out)
        return metric_out
    
    def forward(self, data, pred_y, truth_y, p, c_epoch):
        if self.stdrop ==1:
            distance = self.get_distance(data)
            score = self.get_score(distance)
            total_score = score.clone()
            score = score.argsort(dim=1,descending=False).argsort(dim=1,descending=False)
            score[score<score.shape[1]*p] = -1
            score[score>-1]= 0
            #score = score/p
            score = score.view(data.shape[0], 1, 1, data.shape[3])#.repeat(1, data.shape[1], 1, 1)
            data = data * score * -1
        else:
            if c_epoch > self.s_epoch and c_epoch <= (self.s_epoch+self.cl_epoch) :
                a_epoch = c_epoch - self.s_epoch
                total_score = self.get_quantile_score(pred_y, truth_y, self.sort_rank, a_epoch, self.cl_epoch, qs=[0.1,0.5,0.9])
                #print('metric_score shape is ',metric_score.shape)
                # print('total_score  is ',total_score)
                data = data * total_score * -1
            else: 
                total_score = 1
                return data, total_score      
        return data, total_score
def get_quantile_score2(ypred, ytrue, qs=[0.1,0.5,0.9]):
    '''
    Quantile loss version 2
    Args:
    ytrue (batch_size, time, node, channel)
    ypred (batch_size, time, node, num_quantiles)
    return out(batch_size, time, node, num_quantiles)
    '''
#     ytrue[ytrue < 1] = 0
#     ypred[ypred < 1] = 0    
    assert not ytrue.requires_grad
    losses = []
    null_val = 0.0
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        mask = (ytrue != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)    

    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    for i, q in enumerate(qs):
        errors = ytrue[:,:,:,0] - ypred[:,:,:,i]

        loss = torch.max((q - 1) * errors, q * errors)
        loss = loss * mask[:,:,:,0]
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        losses.append(loss.unsqueeze(-1))
        # 
#         print('torch.cat(losses) shape is ',torch.cat(losses).shape)
    metric_out = torch.cat(losses,dim=3).detach()  # [b,t,n,3]
    return metric_out    
def get_quantile_score3(descending, node, a_epoch, scl_epoch, rank):
    # x [b,t,n,c]
    if a_epoch < scl_epoch:
        spatial_rank_dim = int(node * a_epoch / scl_epoch)
        sorted_indices =  np.argsort(rank)   
        if descending:
            sorted_indices = sorted_indices[::-1]
        # print('rank shape  ',rank.shape)
        # print('sort_rank ',sort_rank)
        boarderline = rank[sorted_indices[spatial_rank_dim-1]]
        # print('boarderline ',boarderline)
        bigger_mask = rank>=boarderline.item() if descending else rank>boarderline.item()
        smaller_mask = rank<boarderline.item() if descending else rank<=boarderline.item()
        # print('bigger_mask ',bigger_mask)
        # print('smaller_mask ',smaller_mask) 
        rank[bigger_mask]=-1 if descending else 0
        rank[smaller_mask]=0 if descending else -1     
        # print('rank ',rank)
        metric_out = rank
    else:
        metric_out = -1
    return metric_out

def get_quantile_score4(descending, node, a_epoch, scl_epoch, rank):
    # x [b,t,n,c]
    rank_flat = rank.flatten()
    if a_epoch < scl_epoch:
        spatial_rank_dim = int(node * a_epoch / scl_epoch)
        sorted_indices =  np.argsort(rank_flat)   
        if descending:
            sorted_indices = sorted_indices[::-1]
        # print('rank shape  ',rank.shape)
        # print('sort_rank ',sort_rank)
        boarderline = rank_flat[sorted_indices[spatial_rank_dim-1]]
        # print('boarderline ',boarderline)
        bigger_mask = rank>=boarderline.item() if descending else rank>boarderline.item()
        smaller_mask = rank<boarderline.item() if descending else rank<=boarderline.item()
        # print('bigger_mask ',bigger_mask)
        # print('smaller_mask ',smaller_mask) 
        rank[bigger_mask]=-1 if descending else 0
        rank[smaller_mask]=0 if descending else -1     
        # print('rank ',rank)
        metric_out = rank
        # print('rank shape is ',rank.shape)
    else:
        metric_out = -1
    return metric_out

def get_quantile_score6(descending, node, a_epoch, scl_epoch, rank):
    # x [b,t,n,c]
    # print('rank ',rank)
    if a_epoch < scl_epoch:
        spatial_rank_dim = int(node * a_epoch / scl_epoch)
        sorted_indices =  np.argsort(rank)   
        if descending:
            sorted_indices = sorted_indices[::-1]
        # print('rank shape  ',rank.shape)

        boarderline = rank[sorted_indices[spatial_rank_dim-1]]
        # print('boarderline ',boarderline)
        bigger_mask = np.where(rank >= boarderline.item())  if descending else rank>boarderline.item()
        smaller_mask = np.where(rank < boarderline.item()) if descending else rank<=boarderline.item()
        # bigger_mask = rank>=boarderline.item() if descending else rank>boarderline.item()
        # smaller_mask = rank<boarderline.item() if descending else rank<=boarderline.item()        
        # print('bigger_mask ',bigger_mask)
        # print('smaller_mask ',smaller_mask) 
        rank[bigger_mask]= -1  if descending else boarderline / rank[bigger_mask] * (-1)
        rank[smaller_mask]= boarderline / rank[smaller_mask] * (-1) if descending else -1    
        # print('rank ',rank)
        metric_out = rank
    else:
        metric_out = -1
    return metric_out
def get_quantile_score5(descending, node, a_epoch, scl_epoch, rank):
    # x [b,t,n,c]
    # print('original rank ',rank)
    if a_epoch < scl_epoch:
        spatial_rank_dim = int(node * a_epoch / scl_epoch)
        sorted_indices =  np.argsort(rank)   
        if descending:
            sorted_indices = sorted_indices[::-1]
        # print('rank shape  ',rank.shape)

        boarderline = rank[sorted_indices[spatial_rank_dim-1]]
        # print('boarderline ',boarderline)
        bigger_mask = np.where(rank >= boarderline.item())  if descending else rank>boarderline.item()
        smaller_mask = np.where(rank < boarderline.item()) if descending else rank<=boarderline.item()
        # bigger_mask = rank>=boarderline.item() if descending else rank>boarderline.item()
        # smaller_mask = rank<boarderline.item() if descending else rank<=boarderline.item()        
        # print('bigger_mask ',bigger_mask)
        # print('smaller_mask ',smaller_mask) 
        rank[bigger_mask]= -1  if descending else -1+ (1+boarderline / rank[bigger_mask] * (-1))*0.1
        rank[smaller_mask]= -1 + (1+boarderline / rank[smaller_mask] * (-1))*0.1 if descending else -1    
        # print('rank ',rank.shape)  # (307,)
        metric_out = rank
    else:
        metric_out = -1
    return metric_out
def get_quantile_score33(descending, node, a_epoch, scl_epoch, rank):
    # x [b,t,n,c]
    sorted_indices = None
    if a_epoch < scl_epoch:
        spatial_rank_dim = int(node * a_epoch / scl_epoch)
        sorted_indices =  np.argsort(rank)   
        if descending:
            sorted_indices = sorted_indices[::-1]
        # print('rank shape  ',rank.shape)
        # print('sort_rank ',sort_rank)
        boarderline = rank[sorted_indices[spatial_rank_dim-1]]
        # print('boarderline ',boarderline)
        bigger_mask = rank>=boarderline.item() if descending else rank>boarderline.item()
        smaller_mask = rank<boarderline.item() if descending else rank<=boarderline.item()
        # print('bigger_mask ',bigger_mask)
        # print('smaller_mask ',smaller_mask) 
        rank[bigger_mask]=-1 if descending else 0
        rank[smaller_mask]=0 if descending else -1     
        # print('rank ',rank)
        metric_out = rank
    else:
        metric_out = -1
    return metric_out,sorted_indices
class SDrop7(nn.Module):

    def __init__(self):
        super(SDrop7, self).__init__()
        # self.adj = self.get_neighbor()

    
    def forward(self, data, mask_rank):
        
        #  mask_rank = mask_rank if type(mask_rank)==int else  mask_rank.repeat(data.shape[0], 1, 1, 1).view(data.shape[0],1,1,data.shape[3]) 
        data = data * mask_rank * -1 
        # exit()
        return data, mask_rank