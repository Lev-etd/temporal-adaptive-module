import torch


class Identity(torch.nn.Module):
    @staticmethod
    def forward(ctx, input):
        return input


class SegmentConsensus(torch.autograd.Function):  
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.consensus_type = consensus_type
        ctx.dim = 1
        ctx.shape = input_tensor.size()
        if ctx.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif ctx.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(ctx.shape[ctx.dim])
        elif ctx.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in, None


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim
    
    @staticmethod
    def forward(ctx, input):
        return SegmentConsensus.apply(input,(self.consensus_type, self.dim))
