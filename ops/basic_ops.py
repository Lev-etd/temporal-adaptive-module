import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)
    
    
    
    
    
    
    

# import torch


# class Identity(torch.nn.Module):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return input


# class SegmentConsensus(torch.autograd.Function):  
#     @staticmethod
#     def forward(ctx, input_tensor):
#         ctx.consensus_type = consensus_type
#         ctx.dim = dim
#         ctx.shape = input_tensor.size()
#         ctx.save_for_backward(consensus_type, dim, shape)
#         if ctx.consensus_type == 'avg':
#             output = input_tensor.mean(dim=ctx.dim, keepdim=True)
#         elif ctx.consensus_type == 'identity':
#             output = input_tensor
#         else:
#             output = None

#         return output

    
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_output, consensus_type, dim, shpe  = ctx.saved_tensors
#         if ctx.consensus_type == 'avg':
#             grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
#         elif ctx.consensus_type == 'identity':
#             grad_in = grad_output
#         else:
#             grad_in = None

#         return grad_in, None


# class ConsensusModule(torch.nn.Module):    
    
#     def __init__(self, consensus_type, dim=1):
#         super(ConsensusModule, self).__init__()
#         self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
#         self.dim = dim
        
    
#     def forward(self, input):
#         return SegmentConsensus(self.consensus_type, self.dim)(input)
# #     @staticmethod
# #     def forward(self, input):
# #         ctx.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
# #         ctx.dim = dim
# #         ctx.save_for_backward(input, consensus_type, dim)
# #         return SegmentConsensus.apply(input,(ctx.consensus_type, ctx.dim))
