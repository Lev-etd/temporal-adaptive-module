import torch


class Identity(torch.nn.Module):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input


class SegmentConsensus(torch.autograd.Function):  
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.consensus_type = consensus_type
        ctx.dim = dim
        ctx.shape = input_tensor.size()
        ctx.save_for_backward(consensus_type, dim, shape)
        if ctx.consensus_type == 'avg':
            output = input_tensor.mean(dim=ctx.dim, keepdim=True)
        elif ctx.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.consensus_type == 'avg':
            grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
        elif ctx.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in, None


class ConsensusModule(torch.nn.Module):

    
    @staticmethod
    def forward(ctx, input):
        ctx.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        ctx.dim = dim
        ctx.save_for_backward(input, consensus_type, dim)
        return SegmentConsensus.apply(input,(ctx.consensus_type, ctx.dim))
