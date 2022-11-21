import os, sys
root_path = '/Users/philfh/code_repo/Utils/'
# sys.path.append(root_path)
os.environ['PYTHONPATH'] = root_path

if __name__ == '__main__':
    a = sys.path
    for i in a:
        print(i)

    from sim_corr_rand.corr_spot_vol import main
    main()