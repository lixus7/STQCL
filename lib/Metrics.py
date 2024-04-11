import numpy as np
def evaluate(y_true, y_pred,qs):
    '''
    Args:
    ytrue (b,t,n,c=1)
    ypred (b,t,n,c=3)
    '''
    rmse,mae,mape,smape,rse,quantiles = np.zeros(y_true.shape[1]),np.zeros(y_true.shape[1]),np.zeros(y_true.shape[1]), \
np.zeros(y_true.shape[1]),np.zeros(y_true.shape[1]),np.zeros((y_true.shape[1],3))
    for t in range(y_true.shape[1]):
        rmse[t] = RMSE(y_true[:,t,:,0], y_pred[:,t,:,1])
        mae[t] = MAE(y_true[:,t,:,0], y_pred[:,t,:,1])
        mape[t] = MAPE(y_true[:,t,:,0], y_pred[:,t,:,1])
        smape[t] = SMAPE(y_true[:,t,:,0], y_pred[:,t,:,1])
        rse[t] = RSE(y_true[:,t,:,0], y_pred[:,t,:,1])
        L_, quantiles[t] = quantile_loss(y_true[:,t,:,0], y_pred[:,t,:,:len(qs)],qs)
    return rmse,mae,mape,smape,rse,quantiles # [T]


def point_evaluate(y_true, y_pred):
    rmse, mae, mape, smape, rse = np.zeros(y_true.shape[1]), np.zeros(y_true.shape[1]), np.zeros(y_true.shape[1]), np.zeros(y_true.shape[1]), np.zeros(y_true.shape[1])
    for t in range(y_true.shape[1]):
        rmse[t] = RMSE(y_true[:,t,:,0], y_pred[:,t,:,0])
        mae[t] = MAE(y_true[:,t,:,0], y_pred[:,t,:,0])
        mape[t] = MAPE(y_true[:,t,:,0], y_pred[:,t,:,0])
        smape[t] = SMAPE(y_true[:,t,:,0], y_pred[:,t,:,0])
        rse[t] = RSE(y_true[:,t,:,0], y_pred[:,t,:,0])
    return rmse, mae, mape, smape, rse


def evaluate_separates(y_true, y_pred,qs):
    '''
    Args:
    ytrue (b,t,n,c=1)
    ypred (b,t,n,c=4) (quantile + point)
    '''
    rmse, mae, mape, smape, rse, quantiles = np.zeros(y_true.shape[1]), np.zeros(y_true.shape[1]), np.zeros(
        y_true.shape[1]), np.zeros(y_true.shape[1]), np.zeros(y_true.shape[1]), np.zeros((y_true.shape[1], 3))
    for t in range(y_true.shape[1]):
        rmse[t] = RMSE(y_true[:, t, :, 0], y_pred[:, t, :, -1])
        mae[t] = MAE(y_true[:, t, :, 0], y_pred[:, t, :, -1])
        mape[t] = MAPE(y_true[:, t, :, 0], y_pred[:, t, :, -1])
        smape[t] = SMAPE(y_true[:, t, :, 0], y_pred[:, t, :, -1])
        rse[t] = RSE(y_true[:, t, :, 0], y_pred[:, t, :, -1])
        L_, quantiles[t] = quantile_loss(y_true[:, t, :, 0], y_pred[:, t, :, 0:-1], qs)
    return rmse, mae, mape, smape, rse, quantiles  # [T]

def quantile_loss(ytrue, ypred, qs):
    '''
    Quantile loss version 2
    Args:
    ytrue (batch_size, node, channel)
    ypred (batch_size, node, num_quantiles)
    return L(float), out(output_horizon,num_quantiles)
    '''
#     ytrue[ytrue < 1] = 0
#     ypred[ypred < 1] = 0    
    out=[]
    with np.errstate(divide='ignore', invalid='ignore'):    
        for i, q in enumerate(qs):
            mask = np.not_equal(ytrue, 0)
            mask = mask.astype(np.float32)
            mask /= np.mean(mask)
            L = np.zeros_like(ytrue).astype(np.float32)
            yq = ypred[:,:,i]
            diff = ytrue - yq
            L += np.maximum(q * diff.astype(np.float32), (q - 1) * diff.astype(np.float32))
            L = np.nan_to_num(L * mask)
            out.append(L)
        # print(np.array(out).shape)      # (3, 64, 12)
        quantile_out = np.array(out)
        return L.mean(), np.mean(quantile_out,axis=(1,2))

def RSE(ytrue, ypred):
    '''
    Args:
    ytrue (batch_size, output_horizon)
    ypred (batch_size, output_horizon, num_quantiles)
    return rse(output_horizon)
    '''  
#     ytrue[ytrue < 1] = 0
#     ypred[ypred < 1] = 0    
    rse = np.sqrt(np.square(ypred - ytrue).sum()) / \
        np.sqrt(np.square(ytrue - ytrue.mean()).sum())
    return rse

def SMAPE(ytrue, ypred, null_val=0):
    '''
    Args:
    ytrue (batch_size, output_horizon)
    ypred (batch_size, output_horizon, num_quantiles)
    return rse(output_horizon)
    '''          
#     ypred[ypred<10]=0 
#     ytrue[ytrue<10]=0
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(ytrue)
        else:
            mask = np.not_equal(ytrue, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        smape = np.abs(np.divide((ypred - ytrue).astype('float32'), (ytrue + ypred) / 2.))
        smape = np.nan_to_num(mask * smape)
        return np.mean(smape) * 100
    
def MAPE(ytrue, ypred, null_val=0):
    if 307 in ytrue.shape:
        ypred[ypred<10]=0
        ytrue[ytrue<10]=0 
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(ytrue)
        else:
            mask = np.not_equal(ytrue, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide((ypred - ytrue).astype('float32'), ytrue))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def RMSE(y_true, y_pred):
#     y_true[y_true < 1] = 0
#     y_pred[y_pred < 1] = 0
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse
        
def MAE(y_true, y_pred):
#     y_true[y_true < 1] = 0
#     y_pred[y_pred < 1] = 0
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae