import train
import param
from template import scheme_dict, command_dict

def main():
    args = param.run_args()
    name = args.scheme.split('-')[0]
    num = args.scheme.split('-')[1]
    scheme = scheme_dict[name][num]

    dev_cmd = " --device " + args.device
    name_cmd = " --name {}[{}]".format(scheme, args.scheme) 
    command = dev_cmd+name_cmd
    for flag in scheme.split('_'):
        command += ' ' + command_dict[flag]

    train_args = param.train_args(command)

    if args.mode == 'train':
        train.main(train_args)

if __name__ == '__main__':
    main()