import math

from src.ccp.ccp import init_ccp, read_ccp


def main():
    rbin = [.8,1,1.2]
    dbin = [.2,.25,.3]
    v = '3D'

    # US2
    
    for r, d in zip(rbin, dbin):
        inname = 'US_P_' + str(d) + '_empty.pkl'
        name = 'US_P_' + str(d) + '_' + str(r) + '_' + v
        try:
            empty = read_ccp(inname)
            empty.compute_stack(v, geocoords=(25, 53, -130, -55), save=name,
            binrad=r/d)
        except FileNotFoundError:
            init_ccp(d, v, 'P', geocoords=(25, 53, -130, -55),
                compute_stack=True, binrad=r/d,
                save=name, verbose=True)

    # # Tibet
    # for r, d, v in zip(rbin, dbin, vel_model):
    #     if d == .15:
    #         continue
    #     name = 'Tibet_S_' + str(d) + '_' + str(r) + '_' + v 
    #     init_ccp(d, v, 'S', geocoords=(20, 43, 66, 102),
    #          compute_stack=True, binrad=r/d,
    #          save=name, verbose=True)
    
    # Alaska
    for r, d in zip(rbin, dbin):
        inname = 'Alaska_P_' + str(d) + '_empty.pkl'
        name = 'Alaska_P_' + str(d) + '_' + str(r) + '_' + v
        try:
            empty = read_ccp(inname)
            empty.compute_stack(v, geocoords=(47, 74, -180, -121), save=name,
            binrad=r/d)
        except FileNotFoundError:
            init_ccp(d, v, 'P', geocoords=(47, 74, -180, -121),
                compute_stack=True, binrad=r/d,
                save=name, verbose=True)
    
    # # Andes
    # for r, d, v in zip(rbin, dbin, vel_model):
    #     if d == .15:
    #         continue
    #     name = 'SAmerica_S_' + str(d) + '_' + str(r) + '_' + v
    #     init_ccp(d, v, 'S', geocoords=(-56, 11, -93, -34),
    #          compute_stack=True, binrad=r/d,
    #          save=name, verbose=True)


main()
