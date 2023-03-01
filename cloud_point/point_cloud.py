import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from scipy import signal
from matplotlib.widgets import Slider, TextBox

class Point_Cloud():
    def __init__(self, path, frame_size=60, eps=0.1, min_samples=5, max=3, min=0.3):
        self.path = path                    # point cloud data path
        self.frame_size = frame_size        # num of points per frame
        self.data_frames = None             # num_of_frames x 3 x frame_size
        self.data_all = None                # 3 x num_of_points
        self.all_cluster = None             # clusetrs
        self.all_cluster_by_frame = None    # clusters x num_of_frames

        self.parse_data(max, min)
        self.run_DBSCAN(eps=eps, min_samples=min_samples)

    
    def parse_data(self, max, min):
        '''
            parse point cloud data into dicts
                max: max distance in meters to be considered as a valid signal
                min: min distance in meters to be considered as a valid signal
        '''
        # start = time.time()
        # read data from file
        print("Loading Point Cloud Data...")
        with open(self.path, "r") as f:
            dataRaw = f.read()
            f.close()
        
        # parse data
        point_data = {"pointID":[], "x":[], "y":[], "z":[]}
        for data in dataRaw.split("---\n")[:-1]:
            line = data.split("\n")
            if abs(float(line[7].split(": ")[1])) < max and abs(float(line[7].split(": ")[1])) > min:
                # if abs(float(line[8].split(": ")[1])) < max and abs(float(line[8].split(": ")[1])) > min:
                    # if abs(float(line[9].split(": ")[1])) < max and abs(float(line[9].split(": ")[1])) > min:
                        point_data["pointID"].append(float(line[6].split(": ")[1]) )
                        point_data["x"].append(float(line[7].split(": ")[1]) )
                        point_data["y"].append(float(line[8].split(": ")[1]) )
                        point_data["z"].append(float(line[9].split(": ")[1]) )

        for key in point_data.keys():
            print(key, len(point_data[key]))

        # organize data into frames
        data_frames = []
        for f_num in np.arange(0, len(point_data["x"])//self.frame_size):
            s = f_num*self.frame_size
            e = (f_num+1)*self.frame_size
            frame = [point_data["x"][s:e], point_data["y"][s:e], point_data["z"][s:e]]
            data_frames.append(frame)

        self.data_frames = np.array(data_frames)
        self.data_all = np.concatenate(self.data_frames.T, axis=1).T
        
        print("data_frames shape:", np.shape(self.data_frames))
        print("Loading Success!")
        # end = time.time()
        # print("parse:", end - start)
    
    def run_DBSCAN(self, eps=0.1, min_samples=5):
        '''
            run DBSCAN on point cloud to find clusters 
                eps: max distance between two samples to be considered in same cluster
                min_samples: min cluster size
        '''
        # start = time.time()
        self.all_cluster = np.array(DBSCAN(eps=eps, min_samples=min_samples).fit_predict(self.data_all))

        self.all_cluster_by_frame = []
        for i, frame in enumerate(self.data_frames):
            cluster = np.array(DBSCAN(eps=eps, min_samples=min_samples).fit_predict(frame.T))
            self.all_cluster_by_frame.append(cluster)
        print("Clustering Success!")
        # end = time.time()
        # print("DBSCAN:", end - start)
        

    def get_cluster_items(self, cluster_idx=0, frame_id=None):
        '''
            get specific cluster by id and/or by frame id
        '''
        if frame_id != None:
            idx = np.argwhere(self.all_cluster_by_frame[frame_id] == cluster_idx).T
            return (self.data_frames[frame_id]).T[idx][0]
        else:
            idx = np.argwhere(self.all_cluster == cluster_idx).T
            return self.data_all[idx][0]
    
    def get_cluster(self, frame_id=None):
        '''
            get all cluster of a frame
        '''
        out_cluster = None
        if frame_id != None:
            out_cluster = self.all_cluster_by_frame[frame_id]
        else:
            out_cluster = self.all_cluster
        return out_cluster

    def label2color(self, labels):
        ''' 
            assign color labels to clusters
        '''
        color = ["b", "g", "r", "c", "m", "y", "k", "silver"]
        out = []
        for i,l in enumerate(labels):
            out.append(color[l])
        return out
    
    def get_label_analysis(self, lables):
        ''' 
            find number of points in a cluster group by colors
        '''
        num = np.max(lables) + 1
        print("\tnoises: ",  len(np.argwhere(lables == -1)), len(np.argwhere(lables == -1))/len(lables))

        color = ["b", "g", "r", "c", "m", "y", "k", "silver"]
        for i in range(num):
            print("\t", i, color[i], len(np.argwhere(lables == i)), len(np.argwhere(lables == i))/len(lables)) 
    
    def plot_cluster(self, cluster_frame, frame_id, cluster_id):
        ''' 
            plot a cluster given cluster id and frame id
        '''
        # fig = plt.figure(figsize=plt.figaspect(0.5))
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        point_scatter = ax.scatter3D(cluster_frame[0], cluster_frame[1], cluster_frame[2], cmap="Greens")
        # ax.scatter3D([0], [0], [0], linewidths = 10, c = "r")

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Clsuter {} in frame {}'.format(cluster_id, frame_id))

        ax.view_init(elev=30, azim=135)

        ax = fig.add_subplot(1, 2, 2)
        ax.scatter(cluster_frame[1], cluster_frame[2])
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        plt.tight_layout()
        

    def plot_points(self, frame_id=0, cluster_id=0, max=None, min=None, remove_ids=None):
        ''' 
            plot all clusters in a given frame 
                max: max distance
                min: min distance
                remove_id: cluster to remove by id
        '''
        # start = time.time()
        clustering = self.get_cluster(frame_id)

        colors = self.label2color(clustering)
        cur_cluster = self.data_frames[frame_id]
        remaining_clusters = clustering
        if remove_ids != None:
            remain_idxs = []
            for idx, c in enumerate(clustering):
                if c not in remove_ids:
                    remain_idxs.append(idx)
            cur_cluster = cur_cluster.T[remain_idxs].T
            colors = np.array(colors).T[remain_idxs].T
            remaining_clusters = np.array(remaining_clusters).T[remain_idxs].T

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(cur_cluster[0], cur_cluster[1], cur_cluster[2], c=colors)
        ax.scatter3D([0], [0], [0], linewidths = 10, c = "r")
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title("All clusters of frame {}".format(frame_id))

        ax.view_init(elev=30, azim=135)
        

        print("Cluster Pred: ")
        self.get_label_analysis(clustering)

        cluster_idxs = np.argwhere(remaining_clusters == cluster_id).flatten()
        self.plot_cluster(cur_cluster.T[cluster_idxs].T, frame_id, cluster_id)

        # end = time.time()
        # print("plot:", end - start)
    
    def plot_interactive(self, frame_id=0):
        def set_axis(frame_id):
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim(-1,3)
            ax.set_ylim(-3,3)
            ax.set_zlim(-3,3)
            ax.set_title("All clusters of frame {}".format(frame_id))
        
        clustering = self.get_cluster(frame_id)
        colors = self.label2color(clustering)
        cur_cluster = self.data_frames[frame_id]

        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes(projection='3d')
        set_axis(frame_id)
        ax.scatter3D(cur_cluster[0], cur_cluster[1], cur_cluster[2], c=colors)
        ax.scatter3D([0], [0], [0], linewidths = 10, c = "r")
        

        ax_ts = plt.axes([0.25, 0.05, 0.40, 0.03])
        ts_num = Slider(ax_ts, 'frame_id', 0, int(len(cur_cluster.T)), 1)

        ax.view_init(elev=30, azim=135)

        def get_num_slider(val):
            val = int(val)
            ax.clear()
            clustering = self.get_cluster(val)
            colors = self.label2color(clustering)
            cur_cluster = self.data_frames[val]
            set_axis(val)
            ax.scatter3D(cur_cluster[0], cur_cluster[1], cur_cluster[2], c=colors)
            ax.scatter3D([0], [0], [0], linewidths = 10, c = "r")


        ts_num.on_changed(get_num_slider)
        plt.show()