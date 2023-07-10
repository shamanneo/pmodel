from torch import optim 

def build_optimizer(args, model) :
    optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9, weight_decay = 5e-4)
    return optimizer
