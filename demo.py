from model import StereoSiamNet

if __name__ == '__main__':
    net_ = StereoSiamNet()
    print(net_.dump_info())
    print('net_built')