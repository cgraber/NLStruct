import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d

import os
import numpy as np

def save_data(path, *vals):
    with open(path, "w") as fout:
        for val in zip(*vals):
            fout.write(",".join([str(item) for item in val]))
            fout.write("\n")

def graph_results(dir_name, model_name, return_vals):
    path = os.path.join(dir_name, 'training_%s.pdf'%(model_name))
    train_vals = return_vals['train_vals']
    train_vals = list(zip(*train_vals))
    if len(train_vals) > 0:
        plt.figure(1)
        plt.clf()
        plt.plot(train_vals[0], train_vals[1])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(path)
        path = os.path.join(dir_name, 'training_%s.png'%(model_name))
        plt.savefig(path)
        path = os.path.join(dir_name, 'training_%s.csv'%(model_name))
        save_data(path, train_vals[0], train_vals[1])

    if 'diff_vals' in return_vals and return_vals['diff_vals'] is not None:
        diff_vals = return_vals['diff_vals']
        diff_vals = sum(diff_vals,[])
        y_updates = return_vals['y_updates']
        y_updates = sum(y_updates,[])
        y_updates = list(zip(*y_updates))
        diff_vals = list(zip(*diff_vals))

        x_axis_vals = list(range(0, 5*len(y_updates[0]), 5))

        plt.figure(2)
        plt.clf()
        plt.plot(x_axis_vals, diff_vals[0], label='T(c,y,w) - T(c,H(x*,c,w),w), Averaged')
        if len(diff_vals) > 2:
            plt.plot(x_axis_vals, diff_vals[3], label='T(c,y,w) - T(c,H(x*,c,w),w), Not Averaged')
        plt.plot(x_axis_vals, [0 for _ in range(len(diff_vals[0]))], 'k--')
        plt.xlabel('Iteration')
        plt.legend()
        path = os.path.join(dir_name, 'topdiff_%s.pdf'%(model_name))
        plt.savefig(path)
        path = os.path.join(dir_name, 'topdiff_%s.png'%(model_name))
        plt.savefig(path)
        path = os.path.join(dir_name, 'topdiff_%s.csv'%(model_name))
        if len(diff_vals) > 2:
            save_data(path, x_axis_vals, diff_vals[0], diff_vals[3])
        else: 
            save_data(path, x_axis_vals, diff_vals[0])

        '''
        plt.figure(2)
        plt.clf()
        plt.plot(x_axis_vals, diff_vals[1], label='lambd^Ty-lambd^TH(x,c,w)')
        plt.plot(x_axis_vals, [0 for _ in xrange(len(diff_vals[0]))], 'k--')
        plt.legend()
        plt.xlabel('Iteration')
        path = os.path.join(dir_name, 'normdiff_%s.pdf'%(model_name))
        plt.savefig(path)
        path = os.path.join(dir_name, 'normdiff_%s.png'%(model_name))
        plt.savefig(path)
        path = os.path.join(dir_name, 'normdiff_%s.csv'%(model_name))
        save_data(path, x_axis_vals, diff_vals[1])
        plt.figure(2)
        '''
        plt.clf()
        plt.plot(x_axis_vals, diff_vals[1], label='lambd^Ty-lambd^TH(x,c,w)')
        plt.plot(x_axis_vals, [0 for _ in range(len(diff_vals[0]))], 'k--')
        plt.legend()
        plt.xlabel('Iteration')
        path = os.path.join(dir_name, 'normdiff_%s.pdf'%(model_name))
        plt.savefig(path)
        path = os.path.join(dir_name, 'normdiff_%s.png'%(model_name))
        plt.savefig(path)
        path = os.path.join(dir_name, 'normdiff_%s.csv'%(model_name))
        save_data(path, x_axis_vals, diff_vals[1])


        if len(diff_vals) > 2:
            plt.figure()
            plt.clf()
            all_data = [x_axis_vals]
            for i in range(len(diff_vals[2][0])):
                data = [val[i] for val in diff_vals[2]]
                all_data.append(data)
                plt.plot(x_axis_vals, data)
            plt.xlabel('Iteration')
            plt.ylabel('y - H(x,c,w)')
            path = os.path.join(dir_name, 'all_topdiffs_%s.pdf'%(model_name))
            plt.savefig(path)
            path = os.path.join(dir_name, 'all_topdiffs_%s.png'%(model_name))
            plt.savefig(path)
            path = os.path.join(dir_name, 'all_topdiffs_%s.csv'%(model_name))
            save_data(path, *all_data)

            plt.figure()
            plt.clf()
            plt.plot(x_axis_vals, diff_vals[4], label='Objective')
            plt.xlabel('Iteration')
            plt.ylabel('Objective')
            path = os.path.join(dir_name, 'objvals_%s.pdf'%(model_name))
            plt.savefig(path)
            path = os.path.join(dir_name, 'objvals_%s.png'%(model_name))
            plt.savefig(path)
            path = os.path.join(dir_name, 'objvals_%s.csv'%(model_name))
            save_data(path, x_axis_vals, diff_vals[4])


        plt.figure(3)
        plt.clf()
        plt.plot(x_axis_vals, y_updates[1], label='Inf Obj Val')
        #plt.plot(range(len(y_updates[0])), y_updates[1], label='||y-H(x)||')
        plt.legend() 
        plt.xlabel('Iteration')
        path = os.path.join(dir_name, 'yvals_%s.pdf'%(model_name))
        plt.savefig(path)
        path = os.path.join(dir_name, 'yvals_%s.png'%(model_name))
        plt.savefig(path)
        path = os.path.join(dir_name, 'yvals_%s.csv'%(model_name))
        save_data(path, x_axis_vals, y_updates[1])#, y_updates[1])
        #plt.figure(4)
        #plt.clf()
        #plt.xlabel('Iteration')
        #plt.plot(range(len(y_updates[0])), y_updates[2])
        #plt.ylabel('||w-lambda||')
        #path = os.path.join(dir_name, '%s_%s_wdiff.pdf'%(dataset_name, model_name))
        #plt.savefig(path)
        cmap = cm.jet
        c = np.linspace(0,10, len(y_updates[2]))
        fig = plt.figure(4)
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        if len(y_updates) > 5:
            ax.scatter(y_updates[5], y_updates[6], y_updates[1], c=c, cmap=cmap)
        else:
            ax.scatter(y_updates[3], y_updates[4], y_updates[1], c=c, cmap=cmap)
        ax.set_xlabel('lambda')
        ax.set_ylabel('y')
        ax.set_zlabel('Obj Value')
        path = os.path.join(dir_name, 'lvy_%s.pdf'%(model_name))
        fig.savefig(path)
        path = os.path.join(dir_name, 'lvy_%s.png'%(model_name))
        fig.savefig(path)
        path = os.path.join(dir_name, 'lvy_%s.csv'%(model_name))
        if len(y_updates) > 5:
            save_data(path, y_updates[5], y_updates[6], y_updates[1]) 
        else:
            save_data(path, y_updates[3], y_updates[4], y_updates[1])

        fig = plt.figure(5)
        fig.clf()
        if len(y_updates) > 5:
            plt.scatter(y_updates[5], y_updates[6], c=c, cmap=cmap)
        else:
            plt.scatter(y_updates[3], y_updates[4], c=c, cmap=cmap)
        plt.xlabel('lambda')
        plt.ylabel('y')
        path = os.path.join(dir_name, 'lvy2d_%s.pdf'%(model_name))
        plt.savefig(path)
        path = os.path.join(dir_name, 'lvy2d_%s.png'%(model_name))
        plt.savefig(path)

        if len(y_updates) > 5:
            plt.figure()
            plt.clf()
            plt.plot(x_axis_vals, y_updates[6], label='Averaged Y')
            plt.plot(x_axis_vals, y_updates[4], label='Y')
            plt.legend()
            plt.xlabel('Iteration')
            path = os.path.join(dir_name, 'avg_y_%s.pdf'%(model_name))
            plt.savefig(path)
            path = os.path.join(dir_name, 'avg_y_%s.png'%(model_name))
            plt.savefig(path)
            path = path[:-3] + 'csv'
            save_data(path, x_axis_vals, y_updates[6], y_updates[4])

            plt.figure()
            plt.clf()
            plt.plot(x_axis_vals, y_updates[5], label='Averaged Lambda')
            plt.plot(x_axis_vals, y_updates[3], label='Lambda')
            plt.legend()
            plt.xlabel('Iteration')
            path = os.path.join(dir_name, 'avg_lambda_%s.pdf'%(model_name))
            plt.savefig(path)
            path = os.path.join(dir_name, 'avg_lambda_%s.png'%(model_name))
            plt.savefig(path)
            path = path[:-3] + 'csv'
            save_data(path, x_axis_vals, y_updates[5], y_updates[3])

    else:
        y_updates = return_vals['y_updates']
        y_updates = sum(y_updates,[])
        plt.figure()
        plt.clf()
        plt.plot(list(range(len(y_updates))), y_updates)
        plt.xlabel('Inference iteration')
        plt.ylabel('Inference objective')
        path = os.path.join(dir_name, 'yvals_%s.pdf'%(model_name))
        plt.savefig(path)
        path = os.path.join(dir_name, 'yvals_%s.png'%(model_name))
        plt.savefig(path)
        path = os.path.join(dir_name, 'yvals_%s.csv'%(model_name))
        save_data(path, list(range(len(y_updates))), y_updates)

