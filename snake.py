import numpy as np
import skimage.draw
from scipy import interpolate
import matplotlib.pyplot as plt


class snake:

    def __init__(self, n_points, im,tau = 200, alpha=0.5, beta=0.5):   
        self.n_points = n_points
        self.points = np.empty((n_points, 2))
        self.prev_points = np.empty((n_points, 2))
        self.normals = np.empty((n_points, 2))
        self.im_values = np.zeros((n_points,1)) 
        self.im = im
        self.Y = im.shape[0]
        self.X = im.shape[1]
        self.f_ext= np.ones((n_points,1))
        self.tau = tau
        self.cycle = 0

        self.init_interp_function()
        self.create_smoothing_matrix(alpha=alpha,beta=beta)

        self.init_snake_to_image()

        self.update_snake(False)
        

    def init_snake(self):
        angs = np.linspace(0,2*np.pi, self.n_points,endpoint=False)
        for i in range(self.n_points):
            self.points[i,:] = [1+np.cos(angs[i]), 1+np.sin(angs[i])]

        self.calc_normals()
    
    def init_snake_to_image(self,r=None):
        x,y = self.im.shape # changes this to self.x ..
        if r is None:
            r = x/np.sqrt(2*np.pi)
            
        angs = np.linspace(0,2*np.pi, self.n_points,endpoint=False)
        for i in range(self.n_points):
            self.points[i,:] = [x/2+np.cos(angs[i])*r, y/2+np.sin(angs[i])*r]

        self.calc_normals()

    def init_interp_function(self):
        X = np.arange(0,self.X)
        Y = np.arange(0,self.Y)
        self.interp_f = interpolate.interp2d(Y,X, self.im.T, kind="linear")

        

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


    def calc_area_means(self):
        inside_mask = skimage.draw.polygon2mask(self.im.shape, self.points)
        outside_mask =  ~inside_mask

        self.m_in = np.mean(self.im[inside_mask])
        self.m_out = np.mean(self.im[outside_mask])
        #print(self.m_in,self.m_out)
    


    def calc_norm_forces(self):
        self.f_ext = (self.m_in - self.m_out)*(2*self.im_values - self.m_in - self.m_out)
        #print(self.f_ext)
    

        
    def constrain_to_im(self):
        self.points[:,0].clip(0,self.X)
        self.points[:,1].clip(0,self.Y)

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
        self.calc_norm_forces()
        

        if update:
            self.prev_points = self.points
            if smoothing:
                self.points = self.smoothing_matrix @( self.points +  self.tau * np.diag(self.f_ext.flatten()) @ self.normals ) 
            else:
                self.points = ( self.points +  self.tau * np.diag(self.f_ext.flatten()) @ self.normals ) 

            self.constrain_to_im()
            
            self.distribute_points()
            if self.cycle % 1 == 0: # only do every t'th cycle to save perfermance?
                self.remove_intersections()
                #self.cycle = 0
            self.cycle += 1

            
        
    def converge_to_shape(self,ax=None, conv_lim_pix=0.1, plot=True, show_normals=False):
        def pop_push(arr, val):
            arr = np.roll(arr, -1)
            arr[-1] = val
            return arr

        self.update_snake(False) # update all values without updating the snake
        if ax is None:
            fig, ax = plt.subplots(2)
        # need better convergence criteria,  i.e. movement of points?
        # lower tau if it bounces?
        last_movement = np.full(7,np.nan)

        while (div := (abs(np.mean(self.im_values) - np.mean([self.m_in,self.m_out] ))/np.mean([self.m_in,self.m_out]) )*100)  > conv_lim_pix:
            movement = np.mean(np.linalg.norm(self.points - self.prev_points, axis=1)**2)
            last_movement = pop_push(last_movement, movement)
            mean_last_movement = np.nanmean(last_movement)
            if plot and self.cycle % 1 == 0: # only plot every t cycles?
                ax[0].clear()
                # ax[1].clear()
                self.show(ax=ax[0],show_normals=show_normals)
                print(self.cycle)
                # ax[1].plot(np.arange(0, self.cycle), movement)
                # ax[1].axhline( y=mean_last_movement)
                plt.draw()
                plt.pause(0.000001)
            self.update_snake()
            # print(div)
            # print(np.sum(self.f_ext))
            print(movement, mean_last_movement, abs((movement-mean_last_movement)/mean_last_movement*100), sep = "\t")
            # print(mean_last_movement)
            # print(movement < mean_last_movement)
        # print(div)



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
        if show_normals:
            ax.plot(self.points[:,0], self.points[:,1],'.-', color="C2")
            self.show_normals(ax=ax)
        else:
            ax.plot(self.points[:,0], self.points[:,1],'-', color="C2")

    def show_normals(self, ax):
        
        adjusted_normals =  self.tau*np.diag(self.f_ext.flatten()) @ self.normals
        ax.quiver(self.points[:,0],self.points[:,1],adjusted_normals[:,0], -adjusted_normals[:,1], color="red")
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