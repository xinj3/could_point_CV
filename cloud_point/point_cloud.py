import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from scipy import signal

class Point_Cloud():
    def __init__(self, path, frame_size=60, eps=0.1, min_samples=5):
        self.path = path
        self.frame_size = frame_size
        self.data_frames = None
        self.data_all = None
        self.all_cluster = None
        self.all_cluster_by_frame = None

        self.parse_data()
        self.run_DBSCAN(eps=eps, min_samples=min_samples)

        

    
    def parse_data(self, max=3, min=0.3):
        print("Loading Point Cloud Data...")
        with open(self.path, "r") as f:
            dataRaw = f.read()
            f.close()
        point_data = {"pointID":[], "x":[], "y":[], "z":[]}

        for data in dataRaw.split("---\n")[:-1]:
            line = data.split("\n")
            if np.float(line[7].split(": ")[1]) < max and np.float(line[7].split(": ")[1]) > min:
                point_data["pointID"].append(np.float(line[6].split(": ")[1]) )
                point_data["x"].append(np.float(line[7].split(": ")[1]) )
                point_data["y"].append(np.float(line[8].split(": ")[1]) )
                point_data["z"].append(np.float(line[9].split(": ")[1]) )

        for key in point_data.keys():
            print(key, len(point_data[key]))

        data_frames = []
        for f_num in np.arange(0, len(point_data["x"])//self.frame_size):
            s = f_num*self.frame_size
            e = (f_num+1)*self.frame_size
            frame = [point_data["x"][s:e], point_data["y"][s:e], point_data["z"][s:e]]
            data_frames.append(frame)

        self.data_frames = np.array(data_frames)
        print("data_frames shape:", np.shape(self.data_frames))
        print("Loading Success!")
    
    def run_DBSCAN(self, eps=0.1, min_samples=5):
        self.data_all = np.concatenate(self.data_frames.T, axis=1).T
        self.all_cluster = np.array(DBSCAN(eps=eps, min_samples=min_samples).fit_predict(self.data_all))

        self.all_cluster_by_frame = []
        for i, frame in enumerate(self.data_frames):
            cluster = np.array(DBSCAN(eps=eps, min_samples=min_samples).fit_predict(frame.T))
            self.all_cluster_by_frame.append(cluster)
        print("Clustering Success!")
        

    def get_cluster_items(self, cluster_idx=0, frame_id=None):
        if frame_id != None:
            idx = np.argwhere(self.all_cluster_by_frame[frame_id] == cluster_idx).T
            return self.data_frames[frame_id][idx][0]
        else:
            idx = np.argwhere(self.all_cluster == cluster_idx).T
            return self.data_all[idx][0]
    
    def get_cluster(self, frame_id=None):
        out_cluster = None
        if frame_id != None:
            out_cluster = self.all_cluster_by_frame[frame_id]
        else:
            out_cluster = self.all_cluster
        return out_cluster

    def label2color(self, labels):
        color = ["b", "g", "r", "c", "m", "y", "k", "w"]
        out = []
        for i,l in enumerate(labels):
            out.append(color[l])
        return out
    
    def get_label_analysis(self, lables):
        num = np.max(lables) + 1
        print("noises: ",  len(np.argwhere(lables == -1)), len(np.argwhere(lables == -1))/len(lables))

        color = ["b", "g", "r", "c", "m", "y", "k", "w"]
        for i in range(num):
            print(i, color[i], len(np.argwhere(lables == i)), len(np.argwhere(lables == i))/len(lables)) 
    
    def plot_cluster(self, cluster_idx, frame_id):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        point_scatter = ax.scatter3D(self.data_frames[frame_id][0][cluster_idx], self.data_frames[frame_id][1][cluster_idx], self.data_frames[frame_id][2][cluster_idx], cmap="Greens")
        # ax.scatter3D([0], [0], [0], linewidths = 10, c = "r")

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.view_init(elev=30, azim=135)

        fig = plt.figure()
        plt.scatter(self.data_frames[frame_id][1][cluster_idx], self.data_frames[frame_id][2][cluster_idx])
        plt.xlabel("y")
        plt.ylabel("z")

    def plot_points(self, frame_id=0, cluster_idx=0):
        clustering = self.get_cluster(frame_id)

        colors = self.label2color(clustering)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.data_frames[frame_id][0], self.data_frames[frame_id][1], self.data_frames[frame_id][2], c=colors)
        ax.scatter3D([0], [0], [0], linewidths = 10, c = "r")

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.view_init(elev=30, azim=135)
        

        print("Cluster Pred: ")
        self.get_label_analysis(clustering)

        self.plot_cluster(np.argwhere(clustering == cluster_idx), frame_id)
    
