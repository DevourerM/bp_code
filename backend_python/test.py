import torch
import torch.nn as nn
import torch.fx

class aaa(nn.Module):
    def __init__(self):
        super(aaa, self).__init__()
        self.fn_Conv2d_1 = nn.Conv2d(bias=True, dilation=1, groups=1, in_channels=3, kernel_size=3, out_channels=3, padding=0, padding_mode='zeros', stride=1)

    def forward(self, x_0):
        v_input_1 = x_0
        v_Conv2d_1 = self.fn_Conv2d_1(v_input_1)
        return v_Conv2d_1
    
model = aaa()
traced_model = torch.fx.symbolic_trace(model)
out = torch.fx.graph_module.print_readable(model)
print(out)