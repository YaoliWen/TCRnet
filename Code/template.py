# 数据集
dataset_dic = {
    'RAF0':'--dataset RAF-DB --subset base',
    'RAFf':'--dataset RAF-DB --subset flip',
}

# 模型
model_cmd = {
    'base':"--model TCR -trans 0 -n_b 2 -pt gap",

    'Tn0a': "--model TCR -trans 0 -n_b 2 -pt avg",
    'Tn1a': "--model TCR -trans 3 -n_b 2 -pt avg",
    'Tn2a': "--model TCR -trans 23 -n_b 2 -pt avg",
    'Tn3a': "--model TCR -trans 123 -n_b 2 -pt avg",

    'Tn0v': "--model TCR -trans 0 -n_b 2 -pt vit",
    'Tn1v': "--model TCR -trans 3 -n_b 2 -pt vit",
    'Tn2v': "--model TCR -trans 23 -n_b 2 -pt vit",
    'Tn3v': "--model TCR -trans 123 -n_b 2 -pt vit",

    'T0a': "--model TCR -trans 0 -n_b 2 -pt avg",
    'T1a': "--model TCR -trans 1 -n_b 2 -pt avg",
    'T2a': "--model TCR -trans 2 -n_b 2 -pt avg",
    'T3a': "--model TCR -trans 3 -n_b 2 -pt avg",

    'T0v': "--model TCR -trans 0 -n_b 2 -pt vit",
    'T1v': "--model TCR -trans 1 -n_b 2 -pt vit",
    'T2v': "--model TCR -trans 2 -n_b 2 -pt vit",
    'T3v': "--model TCR -trans 3 -n_b 2 -pt vit",
    

    'Trn0a': "--model TCR -trans 0 -n_b 2 -pt avg -res",
    'Trn1a': "--model TCR -trans 3 -n_b 2 -pt avg -res",
    'Trn2a': "--model TCR -trans 23 -n_b 2 -pt avg -res",
    'Trn3a': "--model TCR -trans 123 -n_b 2 -pt avg -res",

    'Trn0v': "--model TCR -trans 0 -n_b 2 -pt vit -res",
    'Trn1v': "--model TCR -trans 3 -n_b 2 -pt vit -res",
    'Trn2v': "--model TCR -trans 23 -n_b 2 -pt vit -res",
    'Trn3v': "--model TCR -trans 123 -n_b 2 -pt vit -res",

    'T0a': "--model TCR -trans 0 -n_b 2 -pt avg -res",
    'T1a': "--model TCR -trans 1 -n_b 2 -pt avg -res",
    'T2a': "--model TCR -trans 2 -n_b 2 -pt avg -res",
    'T3a': "--model TCR -trans 3 -n_b 2 -pt avg -res",

    'T0v': "--model TCR -trans 0 -n_b 2 -pt vit -res",
    'T1v': "--model TCR -trans 1 -n_b 2 -pt vit -res",
    'T2v': "--model TCR -trans 2 -n_b 2 -pt vit -res",
    'T3v': "--model TCR -trans 3 -n_b 2 -pt vit -res",
} 

# encoder layer
encoder_cmd = {
    'e0': "-n_h 8 -dp 0.0 -no_bi", # 
}

# 预训练模型
pretrain_cmd = {
    'p0': "--pretrain  MS_Celeb_1M",
}

# 学习率 & batch size
lr_batch_cmd = {
    'l0': "-b 128 --lr 0.001 -lrr 0 -lrs 0 -lri 0",
}

# 基本参数
basic_cmd = {
    'b0': " --momentum 0.9 --wd 0",
}

# 环境参数
device_cmd = {
    'd0': "--epochs 150 -mi_b 64 -j 8",
}

# 损失参数
loss_cmd = {
    'c0': "-var -vr 1"
}


# 模板dictionary
scheme_dict = {
    'Baseline':{
        '0': 'RAF0_base_e0_p0_l0_b0_d0', # 
        '1': 'RAFf_base_e0_p0_l0_b0_d0', # 
    },

    'TC_Res':{
            '0': 'RAF0_Trn2v_e0_c0_p0_l0_b0_d0', #
            '1': 'RAFf_Trn2v_e0_c0_p0_l0_b0_d0', #  
        },
}

# 命令集dictionary
command_dict = {**dataset_dic, **model_cmd, **encoder_cmd, **pretrain_cmd,
                **lr_batch_cmd, **basic_cmd, **device_cmd, **loss_cmd}

