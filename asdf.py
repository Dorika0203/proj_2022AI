class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.solver_option = dict()
        self.n_step = args.n_step
        self.step_size = (self.integration_time[1] - self.integration_time[0]) / self.n_step
        self.solver_option.update({'step_size': self.step_size})
 

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol, method='euler', options=self.solver_option)
        # out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)

        return out[1]


    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
        
class MLP_layer(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(MLP_layer, self).__init__()
        module = nn.Linear
        self._layer = module(dim_in, int(dim_out/2))


    def forward(self, t, x):
        tt = torch.ones_like(x[:, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)



class ODEfunc(nn.Module):
    def __init__(self):
        super(ODEfunc, self).__init__()
        # self.bn_1 = nn.BatchNorm1d(58)
        self.relu_1 = nn.ReLU(inplace=True)
        self.MLP_layer_1 = MLP_layer(116, 116)

        # self.bn_2 = nn.BatchNorm1d(58)
        self.relu_2 = nn.ReLU(inplace=True)
        self.MLP_layer_2 = MLP_layer(116, 116)

        # self.bn_3 = nn.BatchNorm1d(58)
        self.relu_3 = nn.ReLU(inplace=True)
        self.MLP_layer_3 = MLP_layer(116, 116)
        # self.bn_4 = nn.BatchNorm1d(58)
        #####

    def forward(self, t, x):
        # out = self.bn_1(x)
        # out = self.relu_1(out)
        out = self.relu_1(x)
        out = self.MLP_layer_1(t, out)

        # out = self.bn_2(out)
        out = self.relu_2(out)
        out = self.MLP_layer_2(t, out)

        # out = self.bn_3(out)

        #######
        out = self.relu_3(out)
        out = self.MLP_layer_3(t, out)
        # out = self.bn_4(out)
        ########
        return out