from cv2 import kmeans
import numpy as np
import skimage.draw
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans


class snake:

    def __init__(self, n_points, im, tau = 200, alpha=0.5, beta=0.5):   
        self.n_points = n_points
        self.points = np.empty((n_points, 2))
        self.prev_points = np.empty((n_points, 2))
        self.normals = np.empty((n_points, 2))
        self.im_values = np.zeros((n_points,1)) 
        self.im = im
        self.im_raveled = self.im.ravel()
        self.im_raveled_color = np.reshape(self.im, (-1, 3))


        self.Y = im.shape[0]
        self.X = im.shape[1]
        self.f_ext= np.ones((n_points,1))
        self.tau = tau
        self.tau_init = tau
        self.cycle = 0

        self.init_interp_function()
        self.create_smoothing_matrix(alpha=alpha,beta=beta)

        self.init_snake_to_image(r=None)
        

        self.update_snake(False)
        

    def init_snake(self):
        angs = np.linspace(0,2*np.pi, self.n_points,endpoint=False)
        for i in range(self.n_points):
            self.points[i,:] = [1+np.cos(angs[i]), 1+np.sin(angs[i])]

        self.calc_normals()
        self.calc_im_mask()
    
    def init_snake_to_image(self,r=None):
        x,y = self.im.shape # changes this to self.x ..
        if r is None:
            r = x/np.sqrt(2*np.pi)
            
        angs = np.linspace(0,2*np.pi, self.n_points, endpoint=False)
        for i in range(self.n_points):
            self.points[i,:] = [x/2+np.cos(angs[i])*r, y/2+np.sin(angs[i])*r]

        self.calc_normals()

    def init_interp_function(self):
        X = np.arange(0,self.X)
        Y = np.arange(0,self.Y)
        self.interp_f = interpolate.interp2d(Y,X, self.im.T, kind="linear")

    def init_patch_dict(self, n_dict=10, patch_size=11):
        self.n_dict = n_dict
        dx = self.X//(n_dict+1)
        dy = self.Y//(n_dict+1)
        X = np.linspace(dx,self.X-dx,n_dict).astype(np.int64)
        Y = np.linspace(dy,self.Y-dy,n_dict).astype(np.int64)
        XX, YY = np.meshgrid(X,Y)
        self.XX = XX.ravel()
        self.YY = YY.ravel()
        # plt.figure()
        # plt.imshow(self.im)
        # plt.scatter(XX, YY, color='r')
        self.dict = []
        delta = patch_size//2
        for i in range(len(XX)):
            patch = self.im[YY[i]-delta: YY[i]+delta+1, XX[i]-delta: XX[i]+delta+1 ]
            self.dict.append(patch)

        
        
        # print(XX,YY)

    def plot_patches(self):
        patch_work = []
        patch_row = []
        for i in range(self.n_dict):
            patch_row.append(dict[i])
            if i % self.n_dict == 0:
                patch_work.append(patch_row)
                patch_row = []
        plt.figure()


    def calc_normals(self):
        for j,i in enumerate(range(0,self.n_points*2-1,2)):
            neighbours = np.take(self.points,[[i-2,i-1],[i+2,i+3]],mode="wrap")
            vec = neighbours[1,:]-neighbours[0,:]
            n_vec = [vec[1], -vec[0]] # normal vec in image coord. is [y,-x] 
            # normalize
            self.normals[j,:] = n_vec/np.linalg.norm(n_vec)


    def create_smoothing_matrix(self, alpha=0.5, beta=0.5):
        I = np.eye(self.n_points)
        a1 = np.roll(I,1, axis=1)
        a2 = np.roll(I,-1, axis=1)
        A = a1 + a2 + -2*I

        b1 = -1*np.roll(a1,1, axis=1)
        b2 = -1*np.roll(a2,-1, axis=1)
        B = b1 + b2 + 4*a1 + 4*a2 - 6*I

        self.smoothing_matrix = np.linalg.inv(I - alpha*A - beta*B)
    

    

    def get_point_im_values(self):
        #self.im_values = np.empty((self.n_points,1)) 
        for i in range(self.n_points):
            #self.im_values[i] = self.im[int(self.points[i,1]),int(self.points[i,0])] # input as (y,x)
            self.im_values[i] = self.interp_f(self.points[i,1], self.points[i,0])
        #print(self.im_values)

    def calc_im_mask(self):
        self.inside_mask = skimage.draw.polygon2mask(self.im.shape, self.points)
        self.outside_mask =  ~self.inside_mask

    def calc_area_means(self):
        # inside_mask = skimage.draw.polygon2mask(self.im.shape, self.points)
        # outside_mask =  ~inside_mask
        self.calc_im_mask()

        self.m_in = np.mean(self.im[self.inside_mask])
        self.m_out = np.mean(self.im[self.outside_mask])
        #print(self.m_in,self.m_out)

    def calc_area_histograms(self):
        # inside_mask = skimage.draw.polygon2mask(self.im.shape, self.points)
        # outside_mask =  ~inside_mask

        self.calc_im_mask()

        self.bins = np.arange(0, 257) - 0.5
        self.n_bins = len(self.bins)
        self.hist = np.histogram(self.im, bins=self.bins, density=True)
        self.hist_in = np.histogram(self.im[self.inside_mask], bins=self.bins, density=True)
        self.hist_out = np.histogram(self.im[self.outside_mask], bins=self.bins, density=True)

        # self.hist_diff = np.reshape(self.hist_out[0] - self.hist_in[0],(-1,self.n_bins))
        self.hist_scale = self.hist_out[0] + self.hist_in[0]
        self.p_in = np.divide(self.hist_in[0], self.hist_scale, out=np.zeros_like(self.hist_in[0]), where=self.hist_scale!=0) # nan become zero from division with 0
        self.p_out = np.divide(self.hist_out[0], self.hist_scale, out=np.zeros_like(self.hist_out[0]), where=self.hist_scale!=0)
        self.p_diff = np.reshape(self.p_in - self.p_out,(-1,self.n_bins-1))
        self.p_diff = np.nan_to_num(self.p_diff, 0.0)
        self.interp_prop = interpolate.interp1d(np.arange(0,256), self.p_diff, kind="linear")

    def calc_snake_length(self):
        self.length = np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))


    def plot_histograms(self, with_gaussians=False):
        self.calc_area_histograms()

        fig, ax = plt.subplots(2)
        ax[0].bar(np.arange(256), height=self.hist_in[0], width=1.0, color='r')
        ax[0].bar(np.arange(256), height=self.hist_out[0], width=1.0, color='b')
        # ax[0].plot(self.hist_in[1][1:], self.p_in)
        # ax[0].plot(self.hist_in[1][1:], self.p_out)
        # ax[0].plot(self.hist_out[1][1:], self.p_out,color='b')
        # ax[0].plot(self.hist_out[1][1:], self.hist_out[0],color='b')
        # ax[0].plot(self.hist_out[1][1:], self.hist_out[0] - self.hist_in[0])
        # ax[0].plot(self.hist_out[1][1:], self.hist[0], 'k')
        ax[0].set_xlim([0,255])
        # vmin_max = np.max(abs(self.hist_diff))
        ax[1].imshow(self.p_diff, cmap="bwr", aspect='auto', vmin=-1.0, vmax=1.0)
        ax[1].set_xlim([0,255])
        # print(np.sum(abs(self.hist_out[0] - self.hist_in[0])))
        # ax.plot(self.hist[1][1:], self.hist[0], 'k')

        gauss_plot = norm.pdf(self.gauss_x, self.means, self.std)*self.pi
        gauss_total = np.sum(gauss_plot, axis=0)
        if with_gaussians:
            ax.plot(self.gauss_x, gauss_total)
            for i in range(self.peaks):
                ax.plot(self.gauss_x, gauss_plot[i])

        

        



    def init_EM_gaussians(self, peaks=2, std=10):
        self.peaks = peaks
        self.means = np.reshape( np.linspace(255//(peaks+2), 255-255//(peaks+2),peaks), (peaks,1))
        self.std = np.full((peaks,1), std)
        self.gauss_x = np.arange(0, 255)
        self.pi = np.full((peaks,1), 1/peaks)
        # self.gaussian_funcs = [norm.(self.means[i], self.std[i])*self.pi[i] for i in range(peaks)]
        self.im_raveled_tiled = np.tile(self.im_raveled,(self.peaks, 1))
        self.gaussians = norm.pdf(self.im_raveled_tiled, self.means, self.std)



    def EM_converge(self, iter = 10):
        for _ in range(iter):

            gsum = np.sum(self.gaussians * self.pi, axis = 0)

            weights = np.divide(self.gaussians * self.pi, np.tile(gsum,(self.peaks,1) ))
            weights_sum = np.sum(weights, axis=1)

            self.means = np.reshape(np.sum(np.multiply(weights, self.im_raveled_tiled), axis=1) / weights_sum, (self.peaks, 1))
            self.std = np.reshape(np.sum( np.multiply(weights , np.power(self.im_raveled_tiled - self.means, 2)), axis=1) / weights_sum, (self.peaks, 1))
            self.std = np.sqrt(self.std)

            self.pi = np.reshape(weights_sum / len(self.im_raveled), (self.peaks, 1))

            self.gaussians = norm.pdf( self.im_raveled_tiled, self.means, self.std)




    def calc_norm_forces(self, method="means"):
        # using area means
        if method == "means":
            self.f_ext = (self.m_in - self.m_out)*(2*self.im_values - self.m_in - self.m_out)
        # using pixel probability
        if method == "prob":
            self.f_ext = self.interp_prop(self.im_values)
        #print(self.f_ext)
    

        
    def constrain_to_im(self):
        self.points[:,0] = np.clip(self.points[:,0], 0, self.X)
        self.points[:,1] = np.clip(self.points[:,1], 0, self.Y)

    def distribute_points(self):
        """ Distributes snake points equidistantly."""
        N = self.n_points
        d = np.sqrt(np.sum((np.roll(self.points, -1, axis=0)-self.points)**2, axis=1)) # length of line segments
        cum_d = np.r_[0, np.cumsum(d)] # x
        # print(cum_d)
        out = np.r_[self.points, self.points[0:1,:]] # y
        # print(out)
        f = interpolate.interp1d(cum_d, out.T)
        self.points = (f(sum(d)*np.arange(N)/N)).T
        #print(self.points)



    def update_snake(self, update=True, smoothing=True):
        self.get_point_im_values()
        self.calc_normals()
        self.calc_area_means()
        self.calc_area_histograms()
        self.calc_norm_forces(method="prob")
        self.calc_snake_length()
        

        if update:
            self.prev_points = self.points
            if smoothing:
                self.points = self.smoothing_matrix @( self.points +  self.tau * np.diag(self.f_ext.flatten()) @ self.normals ) 
            else:
                self.points = ( self.points +  self.tau * np.diag(self.f_ext.flatten()) @ self.normals ) 

            
            self.distribute_points()
            if self.cycle % 1 == 0: # only do every t'th cycle to save perfermance?
                self.remove_intersections()
                #self.cycle = 0
            self.cycle += 1

            self.constrain_to_im()
            
        
    def converge_to_shape(self,ax=None, conv_lim_pix=0.1, plot=True, show_normals=False):
        def pop_push(arr, val):
            arr = np.roll(arr, -1)
            arr[-1] = val
            return arr
        
        self.update_snake(False) # update all values without updating the snake
        if ax is None:
            fig, ax = plt.subplots(1,2)
        # need better convergence criteria,  i.e. movement of points?
        # lower tau if it bounces?
        movement_all = []
        length_all = []
        movement = 0
        # last_movement = np.full(7,np.nan)
        min_iter = 10
        last_movement = np.full(min_iter,1.0)
        last_length = np.full(min_iter, self.length)#self.length)
        last_length[0] = 0
        
        # while (div := (abs(np.mean(self.im_values) - np.mean([self.m_in,self.m_out] ))/np.mean([self.m_in,self.m_out]) )*100)  > conv_lim_pix:
        print(abs((np.mean(last_length) - self.length) / self.length * 100))
        while abs((perc_diff := (np.mean(last_length) - self.length) / self.length))*100 > 0.1 or self.cycle < 10:
            last_movement = pop_push(last_movement, movement)
            mean_last_movement = np.nanmean(last_movement)
            if plot and self.cycle % 1 == 0: # only plot every t cycles?
                ax[0].clear()
                ax[1].clear()
                self.show(ax=ax[0], show_normals=show_normals)
                
                # ax[1].plot(np.arange(0, self.cycle), movement_all)
                ax[1].plot(np.arange(0, self.cycle), length_all)
                # ax[1].axhline( y=mean_last_movement,color='k')
                # ax[1].axhline( y=np.mean(movement_all),color='r')
                ax[1].axhline( y=np.mean(length_all), color='b')
                ax[1].axhline( y=np.mean(last_length), color='g')
                ax[1].axvline( x=self.cycle - min_iter, color='k')
                # ax[1].set_xlim([0,25])
                plt.draw()
                plt.pause(0.000001)
            self.update_snake()
            movement = np.mean(np.linalg.norm(self.points - self.prev_points, axis=1))
            movement_all.append(movement)
            last_length = pop_push(last_length, self.length)
            length_all.append(self.length)

            # print(div)
            # print(np.sum(self.f_ext))
            # print(movement, mean_last_movement, abs((movement-mean_last_movement)/mean_last_movement*100),np.mean(movement_all), sep = "\t")
            # print(np.mean(last_length) - np.mean(length_all))
            # self.tau = self.tau_init * abs(mean_last_movement-2)
            # self.tau = self.tau_init * ( 1 + abs(perc_diff)/100)
            # if perc_diff <= 0:
            change_factor = abs(abs(perc_diff) -1)
            self.tau *= change_factor
            self.tau = np.clip(self.tau, 1, 100)
            print(self.cycle)
            print(abs(perc_diff))
            print(change_factor)
            
            print(self.tau)
            # print(mean_last_movement)
            # print(movement < mean_last_movement)
        print(perc_diff)
        


    def update_im(self, im):
        
        self.im = im
        self.Y = im.shape[0]
        self.X = im.shape[1]
        self.init_interp_function() # else the interp2d works from the previuos image
        

    
    
    def remove_intersections(self):
        """ Reorder snake points to remove self-intersections.
            Arguments: snake represented by a 2-by-N array.
            Returns: snake.
        """
        def is_counterclockwise(snake):
            """ Check if points are ordered counterclockwise."""
            return np.dot(snake[0,1:] - snake[0,:-1],
                        snake[1,1:] + snake[1,:-1]) < 0

        def is_crossing(p1, p2, p3, p4):
            """ Check if the line segments (p1, p2) and (p3, p4) cross."""
            crossing = False
            d21 = p2 - p1
            d43 = p4 - p3
            d31 = p3 - p1
            det = d21[0]*d43[1] - d21[1]*d43[0] # Determinant
            if det != 0.0 and d21[0] != 0.0 and d21[1] != 0.0:
                a = d43[0]/d21[0] - d43[1]/d21[1]
                b = d31[1]/d21[1] - d31[0]/d21[0]
                if a != 0.0:
                    u = b/a
                    if d21[0] > 0:
                        t = (d43[0]*u + d31[0])/d21[0]
                    else:
                        t = (d43[1]*u + d31[1])/d21[1]
                    crossing = 0 < u < 1 and 0 < t < 1         
            return crossing



        snake = self.points.T

        pad_snake = np.append(snake, snake[:,0].reshape(2,1), axis=1)
        pad_n = pad_snake.shape[1]
        n = pad_n - 1 
        
        for i in range(pad_n - 3):
            for j in range(i + 2, pad_n - 1):
                pts = pad_snake[:,[i, i + 1, j, j + 1]]
                if is_crossing(pts[:,0], pts[:,1], pts[:,2], pts[:,3]):
                    # Reverse vertices of smallest loop
                    rb = i + 1 # Reverse begin
                    re = j     # Reverse end
                    if j - i > n // 2:
                        # Other loop is smallest
                        rb = j + 1
                        re = i + n                    
                    while rb < re:
                        ia = rb % n
                        rb = rb + 1                    
                        ib = re % n
                        re = re - 1                    
                        pad_snake[:,[ia, ib]] = pad_snake[:,[ib, ia]]                    
                    pad_snake[:,-1] = pad_snake[:,0]                
        snake = pad_snake[:,:-1]
        if is_counterclockwise(snake):
            self.points = snake.T
        else:
            self.points =  np.flip(snake, axis=1).T















    """ PLOTTING """
    def show(self, ax=None, show_normals=False):
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.imshow(self.im,cmap="gray")
        ax.plot(self.points[:,0], self.points[:,1],'-', color="C2")
        if show_normals:
            # ax.plot(self.points[:,0], self.points[:,1],'.-', color="C2")
            self.show_normals(ax=ax)
        # else:

    def show_normals(self, ax):
        
        adjusted_normals =  self.tau * np.diag(self.f_ext.flatten()) @ self.normals
        ax.quiver(self.points[:,0], self.points[:,1], adjusted_normals[:,0], adjusted_normals[:,1], color="red", minshaft=1 ,minlength=0.1, scale=0.1, units="xy", angles="xy")
        #ax.quiver(self.points[:,0],self.points[:,1],self.normals[:,0], -self.normals[:,1], color="green")

    def plot_im_values(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.clear()
        ax.set_ylim([0,1])
        ax.plot(self.im_values, '.-')
        ax.axhline(y = self.m_in, linestyle='--',color="gray",linewidth=0.5)
        ax.axhline(y = self.m_out, linestyle='--',color="gray",linewidth=0.5)
        ax.axhline(y = np.mean(self.im_values), linestyle='--',color="red",linewidth=0.5)
        ax.axhline(y = np.mean([self.m_in,self.m_out]), linestyle='-',color="gray")



    def init_clusters(self, n_cluster = 2):
        self.n_clusters = n_cluster
        self.model = KMeans(n_clusters=self.n_clusters)
        self.label = self.model.labels_


    ### Clusters
    def clustering(self):

        fit = self.model.fit(self.im_raveled_color)
        self.label = fit.labels_

        self.im[self.inside_mask]

        return 

    