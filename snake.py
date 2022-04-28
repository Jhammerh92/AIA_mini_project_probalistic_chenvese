from cv2 import kmeans
import cv2
import numpy as np
import skimage.draw
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.signal as signal
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class snake:

    def __init__(self, n_points, im, tau = 200, alpha=0.05, beta=0.05, r=None, weights = [1/3, 1/3, 1/3], method = "means"):   
        self.n_points = n_points
        self.points = np.empty((n_points, 2))
        self.prev_points = np.empty((n_points, 2))
        self.normals = np.empty((n_points, 2))
        self.im_values = np.zeros((n_points,1)) 
        self.patch_values = np.zeros((n_points,1)) 
        self.im = im

        
        if (im.ndim == 3):
            self.im_color = im
            self.im = cv2.cvtColor(self.im_color, cv2.COLOR_RGB2GRAY) * 255
            self.im_values_color = np.zeros((n_points,3))
        else:
            self.im_color = None
        self.im_raveled = self.im.ravel()
       
        self.weights = np.array([weights])

        self.Y = im.shape[0]
        self.X = im.shape[1]
        XX, YY = np.meshgrid(np.arange(self.X), np.arange(self.Y))
        XX = XX.ravel()
        YY = YY.ravel()
        self.XXYY = np.c_[XX,YY]


        self.f_ext= np.ones((n_points,1))
        self.tau = tau
        self.tau_init = tau
        self.cycle = 0

        self.init_interp_function()
        self.create_smoothing_matrix(alpha=alpha,beta=beta)
        
        self.method = method

        self.init_snake_to_image(r=r)

        self.init_patch_dict(patch_size=11)

        self.update_snake(False)
        

    def init_snake(self):
        angs = np.linspace(0,2*np.pi, self.n_points,endpoint=False)
        for i in range(self.n_points):
            self.points[i,:] = [1+np.cos(angs[i]), 1+np.sin(angs[i])]

        self.calc_normals()
        self.calc_im_mask()
    
    def init_snake_to_image(self,r=None):
        y,x = self.im.shape # changes this to self.x ..
        if r is None:
            r = x/np.sqrt(2*np.pi)
            
        angs = np.linspace(0,2*np.pi, self.n_points, endpoint=False)
        for i in range(self.n_points):
            self.points[i,:] = [x/2+np.cos(angs[i])*r, y/2+np.sin(angs[i])*r]

        self.calc_normals()

    def interp2d_to_points(self, f, x, y):
        # f is the 2d interp function
        return interpolate.dfitpack.bispeu(f.tck[0],
                                            f.tck[1],
                                            f.tck[2], 
                                            f.tck[3], 
                                            f.tck[4], 
                                            x, y)[0]
                                            
    def interp_color(self,x,y):
        R = self.interp2d_to_points(self.interp_color_R, x, y)
        G = self.interp2d_to_points(self.interp_color_G, x, y)
        B = self.interp2d_to_points(self.interp_color_B, x, y)
        # R = self.interp_color_R(x,y)
        # G = self.interp_color_G(x,y)
        # B = self.interp_color_B(x,y)

        return np.c_[R, G, B]


    def init_interp_function(self):
        X = np.arange(0,self.X)
        Y = np.arange(0,self.Y)
        self.interp_f = interpolate.interp2d(Y,X, self.im.T, kind="linear")
        
        if not (self.im_color is None):
            self.interp_color_R = interpolate.interp2d(Y,X, self.im_color[:,:,0].T, kind="linear")
            self.interp_color_G = interpolate.interp2d(Y,X, self.im_color[:,:,1].T, kind="linear")
            self.interp_color_B = interpolate.interp2d(Y,X, self.im_color[:,:,2].T, kind="linear")


    def init_patch_dict(self, n_dict=10, patch_size=11):
        self.n_dict = n_dict
        self.patch_size = patch_size
        self.delta = patch_size//2
        self.padded_im = np.pad(self.im, pad_width=self.delta, constant_values=0)

        dx = self.X//(n_dict+1)
        dy = self.Y//(n_dict+1)
        X = np.linspace(dx,self.X-dx,n_dict).astype(np.int64)
        Y = np.linspace(dy,self.Y-dy,n_dict).astype(np.int64)
        XX, YY = np.meshgrid(X,Y)
        self.XX = XX.ravel()
        self.YY = YY.ravel()
        self.patch_coords = np.c_[self.XX, self.YY]
        # plt.figure()
        # plt.imshow(self.im)
        # plt.scatter(XX, YY, color='r')
        self.dict = []
        self.dict_ravel = []
        for i in range(len(self.XX)):
            patch = self.im[self.YY[i]-self.delta: self.YY[i]+self.delta+1, self.XX[i]-self.delta: self.XX[i]+self.delta+1 ]
            self.dict.append(patch.astype(np.float64))
            self.dict_ravel.append(patch.astype(np.float64).ravel())


        self.init_im_dict()
        self.init_knn_fitter()


    def init_im_dict(self):
        im_dict = []
        for y in range(self.Y):
            y += self.delta
            for x in range(self.X):
                x += self.delta
                im_dict.append(self.padded_im[y-self.delta: y+self.delta+1, x-self.delta: x+self.delta+1].ravel().astype(np.float64))
        self.im_dict = np.vstack(im_dict)



    def least_square_crosscorr(self, patch):
        epsilon = 1e-6
        delta = self.patch_size//2
        response = np.empty_like(self.im.astype(np.float64))
        for y in range(self.Y):
            y += self.delta
            for x in range(self.X):
                x += self.delta
                response[y-self.delta,x-self.delta] = np.sum(abs((self.padded_im[y-self.delta: y+delta+1, x-self.delta: x+self.delta+1] - patch))**2)
        # response = np.log(response + epsilon)
        # print(response.shape)
        # print(self.im.shape)

        return (-response+ np.max(response)+ epsilon)

    # virker ikke med ravelled dicts
    def conv_patch_dict(self):
        # test_patch = 53
        # delta = self.patch_size//2
        stack = []
        for patch_idx in range(self.n_dict**2):
            print(patch_idx)
            patch = self.dict[patch_idx].astype(np.float64)
            # patch = (patch-np.min(patch))/np.max(patch-np.min(patch)) * 255
            patch_mean = np.mean(patch)
            # im_mean = np.mean(self.im.astype(np.float64))
            # im_patch_response = signal.correlate2d(self.im.astype(np.float64)-patch_mean, patch-patch_mean, mode='same', fillvalue = 0.0)
            im_patch_response = self.least_square_crosscorr(patch)
            stack.append(im_patch_response)
            # im_patch_response_test = self.ordered_crosscorr(patch)
            max_response_yx = np.unravel_index(np.argmax(im_patch_response, axis=None), im_patch_response.shape)
        stack = np.dstack(stack)
        print(stack.shape)
        assignment = np.argmax(stack, axis=2)
        print(assignment)
        plt.figure()
        plt.imshow(assignment)

    
    def init_knn_fitter(self):
        self.knn_fitter = NearestNeighbors(n_neighbors=1, radius=self.patch_size * 255.0) # kan måske laves i init
        self.knn_fitter.fit(self.dict_ravel)



    def calc_patch_knn(self):

        # test = knn_fitter.kneighbors(np.random.rand(2,11**2)*255, return_distance=False)
        # this next line is "slow"
        self.dict_assignment = self.knn_fitter.kneighbors(self.im_dict, return_distance=False)

        self.dict_assignment = np.reshape(self.dict_assignment,(self.im.shape))

        self.dict_assignment_in = self.dict_assignment[self.inside_mask]
        self.dict_assignment_out = self.dict_assignment[self.outside_mask]
        self.patch_bins = np.arange(0,101)-0.5
        self.patch_hist = np.histogram(self.dict_assignment, bins=self.patch_bins, density=True)
        self.patch_hist_in = np.histogram(self.dict_assignment[self.inside_mask], bins=self.patch_bins, density=True)
        self.patch_hist_out = np.histogram(self.dict_assignment[self.outside_mask], bins=self.patch_bins, density=True)

        # self.hist_diff = np.reshape(self.hist_out[0] - self.hist_in[0],(-1,self.n_bins))
        self.patch_hist_scale = self.patch_hist_out[0] + self.patch_hist_in[0]
        self.patch_p_in = np.divide(self.patch_hist_in[0], self.patch_hist_scale, out=np.zeros_like(self.patch_hist_in[0]), where=self.patch_hist_scale!=0) # nan become zero from division with 0
        self.patch_p_out = np.divide(self.patch_hist_out[0], self.patch_hist_scale, out=np.zeros_like(self.patch_hist_out[0]), where=self.patch_hist_scale!=0)
        self.patch_p_diff = np.reshape(self.patch_p_in - self.patch_p_out,(-1,len(self.patch_bins)-1))
        self.patch_p_diff = np.nan_to_num(self.patch_p_diff, 0.0)
        self.patch_interp_prop = interpolate.interp1d(np.arange(0,100), self.patch_p_diff, kind="linear")


        # DEBUGGING plots
        # plt.figure()
        # plt.bar(np.arange(100), height=self.patch_hist[0], width=1.0, color='k', alpha=0.5)
        # plt.bar(np.arange(100), height=self.patch_hist_in[0], width=1.0, color='r', alpha=0.5)
        # plt.bar(np.arange(100), height=self.patch_hist_out[0], width=1.0, color='b', alpha=0.5)
        # print(test)

    



    def plot_patch_dict(self):
        patch_work = []
        patch_row = []
        for i in range(len(self.patch_coords)):
            patch_row.append(self.dict[i])
            if (i+1) % (self.n_dict) == 0 and i > 0:
                patch_work.append(np.concatenate(patch_row, axis =1))
                patch_row = []
        fig, ax = plt.subplots(1,2)
        patch_work_array = np.concatenate(patch_work, axis = 0)
        ax[0].imshow(patch_work_array, cmap= "gray")
        ax[1].imshow(self.dict_assignment, cmap='nipy_spectral')

    


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
        # vectorised
        self.im_values = self.interp2d_to_points(self.interp_f,self.points[:,1], self.points[:,0])
        ravel_idx = ((self.Y) * np.floor(self.points[:,1]) + np.floor(self.points[:,0])).astype(np.int64)
        # print(self.XXYY[ravel_idx],  self.points, self.XXYY[ravel_idx]- self.points)
        self.patch_values = self.knn_fitter.kneighbors(np.array(self.im_dict[ravel_idx]), return_distance=False)
        if not np.all(self.im_color is None):
            self.im_values_color = self.interp_color(self.points[:,1], self.points[:,0])
        
        # for i in range(self.n_points):
            # self.im_values[i] = self.im[int(self.points[i,1]),int(self.points[i,0])] # input as (y,x)
            # self.im_values[i] = self.interp_f(self.points[i,1], self.points[i,0])
            # get xy to abs ravel_index
            # ravel_idx = int(np.floor((self.X-1) * self.points[i,0]) + np.floor(self.points[i,1]))
            # self.patch_values[i] = self.knn_fitter.kneighbors(np.array([self.im_dict[ravel_idx]]), return_distance=False)
        #print(self.im_values)
        return

    def calc_im_mask(self):
        self.inside_mask = skimage.draw.polygon2mask(self.im.shape, self.points)
        self.outside_mask =  ~self.inside_mask

    def calc_area_means(self):
        # inside_mask = skimage.draw.polygon2mask(self.im.shape, self.points)
        # outside_mask =  ~inside_mask
        # self.calc_im_mask()

        self.m_in = np.mean(self.im[self.inside_mask])
        self.m_out = np.mean(self.im[self.outside_mask])
        #print(self.m_in,self.m_out)

    def calc_area_histograms(self):
        # inside_mask = skimage.draw.polygon2mask(self.im.shape, self.points)
        # outside_mask =  ~inside_mask
        # self.calc_im_mask()

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


    def plot_patch_histograms(self, ax=None):
        if (ax is None):
            fig, ax = plt.subplots(2)
        ax[0].bar(np.arange(100), height=self.patch_hist_in[0], width=1.0, color='r',alpha=0.5)
        ax[0].bar(np.arange(100), height=self.patch_hist_out[0], width=1.0, color='b',alpha=0.5)
        ax[0].set_xlim([0,100])
        # vmin_max = np.max(abs(self.hist_diff))
        ax[1].imshow(self.patch_p_diff, cmap="bwr", aspect='auto', vmin=-1.0, vmax=1.0)
        ax[1].set_xlim([0,100])




    def plot_histograms(self,ax=None, with_gaussians=False):
        self.calc_area_histograms()
        if (ax is None):
            fig, ax = plt.subplots(2)
        ax[0].bar(np.arange(256), height=self.hist_in[0], width=1.0, color='r',alpha=0.5)
        ax[0].bar(np.arange(256), height=self.hist_out[0], width=1.0, color='b',alpha=0.5)
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

        if with_gaussians:
            gauss_plot = norm.pdf(self.gauss_x, self.means, self.std)*self.pi
            gauss_total = np.sum(gauss_plot, axis=0)
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
            self.f_ext = (self.m_in - self.m_out)*(2*self.im_values - self.m_in - self.m_out).flatten()
        # using pixel probability
        elif method == "prob":
            self.f_ext = self.interp_prop(self.im_values).flatten()

        elif method == "patch_prob":
            # = self.knn_fitter.kneighbors(self.im_dict, return_distance=False)
            self.f_ext = self.patch_interp_prop(self.patch_values).flatten()
        elif method == "cluster_prob":
            

            self.f_ext = -(self.prob_cluster - (1 - self.prob_cluster)).flatten()
        #print(self.f_ext)
        elif method == "unify" :
            forces = np.array([self.interp_prop(self.im_values).flatten(),
                                    self.patch_interp_prop(self.patch_values).flatten(),
                                    (-(self.prob_cluster - (1 - self.prob_cluster))).flatten()]).T

            self.f_ext = np.sum(self.weights * forces, axis=1)

    

        
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
        self.calc_im_mask()
        self.calc_area_means()
        self.calc_area_histograms()
        self.calc_patch_knn()
        self.clustering() 
        self.calc_norm_forces(method=self.method) # forces skal adderes med weights
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
            
        
    def converge_to_shape(self, ax=None, conv_lim_pix=0.1, plot=True, show_normals=False):
        def pop_push(arr, val):
            arr = np.roll(arr, -1)
            arr[-1] = val
            return arr
        
        # self.update_snake(False) # update all values without updating the snake
        if ax is None:
            fig, ax = plt.subplots(1,2)
            # fig_hist, ax_hist = plt.subplots(2,1)
            # fig_patch_hist, ax_patch_hist = plt.subplots(2,1)


        # need better convergence criteria,  i.e. movement of points?
        # lower tau if it bounces?
        movement_all = []
        length_all = []
        movement = 0
        # last_movement = np.full(7,np.nan)
        min_iter = 40
        last_movement = np.full(min_iter,1.0)
        last_length = np.full(min_iter, self.length)#self.length)
        last_length[0] = 0
        
        plot_snake_line, plot_normals = self.show(ax=ax[0], show_normals=show_normals)
        # while (div := (abs(np.mean(self.im_values) - np.mean([self.m_in,self.m_out] ))/np.mean([self.m_in,self.m_out]) )*100)  > conv_lim_pix:
        # print(abs((np.mean(last_length) - self.length) / self.length * 100))
        plt.pause(0.001)
        while abs((perc_diff := (np.mean(last_length) - self.length) / self.length))*100 > 0.1 or self.cycle < 50:
            last_movement = pop_push(last_movement, movement)
            length_all.append(self.length)
            last_length = pop_push(last_length, self.length)
            mean_last_movement = np.nanmean(last_movement)
            if plot and self.cycle % 1 == 0: # only plot every t cycles?
                # ax[0].clear()
                ax[1].clear()
                plot_snake_line[0].set_data(self.points[:,0], self.points[:,1])
                plot_normals.set_offsets(self.points)
                adjusted_normals = self.tau * np.diag(self.f_ext.flatten()) @ self.normals
                plot_normals.set_UVC(adjusted_normals[:,0], adjusted_normals[:,1]) 
                
                # ax[1].plot(np.arange(0, self.cycle), movement_all)
                # ax[1].axhline( y=mean_last_movement,color='k')
                # ax[1].axhline( y=np.mean(movement_all),color='r')

                ax[1].plot(np.arange(0, self.cycle+1), length_all)
                ax[1].axhline( y=np.mean(length_all), color='b')
                ax[1].axhline( y=np.mean(last_length), color='g')
                ax[1].axvline( x=self.cycle - min_iter, color='k')


                # self.plot_histograms(ax=ax_hist)

                # self.plot_patch_histograms(ax=ax_patch_hist)

                # ax[1].set_xlim([0,25])
                # plt.draw()
                fig.canvas.draw_idle()
                fig.canvas.start_event_loop(0.0001)
                # fig.canvas.flush_events()
                # fig_hist.canvas.draw_idle()
                # fig_hist.canvas.flush_events()
                # plt.draw()
                # plt.pause(0.000001)
            self.update_snake()
            movement = np.mean(np.linalg.norm(self.points - self.prev_points, axis=1))
            movement_all.append(movement)

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
            print(abs(perc_diff))
            print(change_factor)
            print(self.tau)
            
            
            
            print(self.cycle)




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
        # ax.imshow(self.im,cmap="gray")
        if (np.any(self.im_color == None)):
            ax.imshow(self.im,cmap="gray")
        else:
            ax.imshow(self.im_color)
        # ax.plot(self.points[:,0], self.points[:,1],'-', color="C2")
        line = ax.plot(self.points[:,0], self.points[:,1],'-', color="C2")

        normals = None
        if show_normals:
            # ax.plot(self.points[:,0], self.points[:,1],'.-', color="C2")
            normals = self.show_normals(ax=ax)
        # else:
        return line, normals

    def show_normals(self, ax):
        
        adjusted_normals =  self.tau * np.diag(self.f_ext.flatten()) @ self.normals
        normals = ax.quiver(self.points[:,0], self.points[:,1], adjusted_normals[:,0], adjusted_normals[:,1], color="red", minshaft=1 ,minlength=0.1, scale=0.1, units="xy", angles="xy")
        #ax.quiver(self.points[:,0],self.points[:,1],self.normals[:,0], -self.normals[:,1], color="green")

        return normals

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



    # def init_clusters(self, n_cluster = 1):
    #     self.n_clusters = n_cluster
    #     self.model = KMeans(n_clusters=self.n_clusters)


    ### Clusters
    def clustering(self):

        if (np.all(self.im_color is None)):
            return

        cluster_in = np.reshape(self.im_color[self.inside_mask, :], (-1, 3))
        cluster_out = np.reshape(self.im_color[self.outside_mask, :], (-1, 3))

        cluster_center_in = np.average(cluster_in, axis = 0)
        cluster_center_out = np.average(cluster_out, axis = 0)

        dist_in = np.linalg.norm(self.im_values_color - cluster_center_in, axis = 1) 
        dist_out = np.linalg.norm(self.im_values_color - cluster_center_out, axis = 1) 
        
        self.prob_cluster = dist_in / (dist_in + dist_out)

        return 
    


    