import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from bezier import bezier
from scipy.spatial import ConvexHull
from io import BytesIO
import base64
import time

class Ellipse:
    def __init__(self, ellipse_params=np.array([1.5,3.5]), speedel = 0.01, t_number=1000):
        self.speed = speedel
        self.params = ellipse_params
        self.ell_flag=0
        self.ellipse = plt.plot([], [], lw=5, c='orchid')[0]
        self.t = np.linspace(0,2*np.pi,t_number)

    def move(self):
        a,b=self.params
        if not self.ell_flag:
            self.params+=[-self.speed,self.speed]
        else:
            self.params += [self.speed, -self.speed]

        if self.params[0] < 1:
            self.ell_flag = 1
        if self.params[1] < 1:
            self.ell_flag = 0
        self.ellipse.set_data(a * np.cos(self.t),b * np.sin(self.t))

    def projection(self,point):
        x, y = point
        a, b = self.params
        if x != 0:
            tg = y / x
            px = a * b / (b * b + a * a * tg * tg) ** 0.5 * np.sign(x)
            py = a * b * tg / (b * b + a * a * tg * tg) ** 0.5 * np.sign(x)
        else:
            px, py = 0, b
        return np.array([px,py])

    def tangent(self,point):
        a, b = self.params
        px, py = self.projection(point)
        kassat = np.array([py / b / b, -px / a / a])
        return kassat / np.linalg.norm(kassat)


#-----------------------------------------------------------------------


class Point:
    def __init__(self, start = np.array([0.5, 0.5]), vector=np.array([-3, 7]), speed=0.1,tail_length=20):
        self.coords = start
        self.vector = vector / np.linalg.norm(vector)
        self.speed = speed
        self.point=plt.plot([], [], marker='*', markersize=9)[0]
        self.next=self.coords+self.speed*self.vector

        self.tail_length = tail_length
        self.tail_coords=[[start[0]],[start[1]]]
        self.tail = plt.plot([], [], '--', lw=5, alpha=0.3)[0]

    def move(self):
        self.coords=self.coords+self.speed*self.vector
        self.next=self.coords+self.speed*self.vector

        self.point.set_data(*self.coords)
        self.point.set_color(tuple(np.random.rand(1,3)[0]))

        self.tail_coords[0].append(self.coords[0])
        self.tail_coords[1].append(self.coords[1])
        if len(self.tail_coords[0])>=self.tail_length:
            self.tail_coords[0].pop(0)
            self.tail_coords[1].pop(0)
        self.tail.set_data(*self.tail_coords)


#-----------------------------------------------------------------------


def animate_ellipse(Npoints=5,speed = 0.01,fps=30):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(7,5))

    #ax=plt.axes(xlim=(-4.5, 4.5), ylim=(-4.5, 4.5))

    ellipse1=Ellipse(speedel=speed/500)
    list_points=[Point(np.array([0.5,0.5]),(np.random.rand(1,2)-np.ones((1,2))/2)[0],speed=speed/50) for i in range(Npoints)]

    def drawframe(iframe):
        ellipse1.move()
        for i in list_points:

            #!
            projection = ellipse1.projection(i.coords)

            if np.linalg.norm(i.next) > np.linalg.norm(projection):
                kassat = ellipse1.tangent(i.coords)
                if np.dot(kassat, i.vector) < 0:
                    kassat = -kassat
                i.vector = 2 * np.dot(i.vector, kassat) * kassat - i.vector

                i.tail.set_color(tuple(np.random.rand(1, 3)[0]))
            # !

            i.point.set_color(tuple(np.random.rand(1, 3)[0]))
            i.move()
        local=[list_points[i].point for i in range(Npoints)]+[list_points[i].tail for i in range(Npoints)]
        return ellipse1.ellipse, local
    anim = FuncAnimation(fig, drawframe,frames=2000, interval=300/fps, blit=False)
    return anim




#-----------------------------------------------------------------------




# np.random.seed(1)
def normalize(vectors):
    return np.array([vector/np.linalg.norm(vector) for vector in vectors])


class Bezier_curve:
    def __init__(self, speed=0.1):
        #first: N random points
        N=20
        loc_points=np.random.rand(N,2)*10-np.ones((N,2))*5

        #we take convex hull and transform hull points to one big closed bezier curve
        #glued by small bezier curves of degree 2
        loc_conv_vert=ConvexHull(loc_points).vertices
        self.conv_points=loc_points[loc_conv_vert,:]

        # loc_mask=np.array([True]*len(loc_points))
        # loc_mask[loc_conv_vert]=False
        # #the inner points of convex hull are supposed to be stars that reflect from walls
        # self.points=loc_points[loc_mask,:]

        #coordinates data for plotting bezier curves
        self.t = np.linspace(0., 1., 100)
        self.list_curves_plt = [plt.plot([], [], '-', lw=5, alpha=1, c='m')[0] for i in range(len(self.conv_points))]
        #list of objects "bezier curve" from module bezier
        self.list_curves=[]

        #we want our bezier curve to move so we map each convex point to its vector=direction
        self.directions=normalize(np.random.rand(self.conv_points.shape[0],2)-np.ones((self.conv_points.shape[0],2))/2)

        #speed of moving bezier curve
        self.speed = speed
        #speed for turning direction vectors (see next method)
        self.speedturn = self.speed*20

    def move(self):
        #make one step for each conv point along its direction
        self.conv_points+=self.directions*self.speed

        #randomly change direction of each conv point to make it move chaotic
        #parameter speedturn helps to adjust smoothness of bezier move.
        #if bezier curve hits the square |x|<5 |y|<5 then change direction to opposite to bounce from square
        for i in range(len(self.conv_points)):
            if abs(self.conv_points[i,0])>5 or abs(self.conv_points[i,1])>5:
                self.directions[i]=[-self.conv_points[i,0],-self.conv_points[i,1]]
            else:
                self.directions[i]=self.directions[i]+(np.random.rand(1,2)-np.ones((1,2))/2)*self.speedturn
                self.directions[i]/=np.linalg.norm(self.directions[i])


        local=np.vstack((self.conv_points,[self.conv_points[0]],[self.conv_points[1]]))
        #here we update our big bezier curve objects corresponding to conv points
        self.list_curves = [bezier.Curve(
            [[(local[i, 0] + local[i + 1, 0]) / 2, local[i + 1, 0], (local[i + 1, 0] + local[i + 2, 0]) / 2],
             [(local[i, 1] + local[i + 1, 1]) / 2, local[i + 1, 1], (local[i + 1, 1] + local[i + 2, 1]) / 2]],
            degree=2) for i in range(len(local) - 2)]

        #here we update our big bezier curve which we will plot
        for i in range(len(self.list_curves)):
            self.list_curves_plt[i].set_data(*self.list_curves[i].evaluate_multi(self.t))


def animate_bezier(N=12,speed=1,fps=30):
    plt.style.use('dark_background')

    fig = plt.figure(figsize=(7,5))
    #here we fix square on fig
    #ax=plt.axes(xlim=(-5, 5), ylim=(-5, 5))

    #initializing the first frame here
    b=Bezier_curve(speed/500)
    ranlist=[normalize(np.random.rand(1,2)-np.ones((1,2))/2)[0] for i in range(N)]
    list_points = [Point(np.array([0,0]),ranlist[i],speed/50) for i in range(N)]

    def drawframe(iframe):
        #move bezier curve
        b.move()
        for i in list_points:

            #we check if the point intersect the bezier curve in 2 or less steps
            #and if intersect then we reflect the direction (it called "vector" for points)
            # of its point by the tangent vector of curve in the intersection point
            line = bezier.Curve([*zip(i.coords, i.next+i.vector*i.speed)], degree=1)
            for j in b.list_curves:
                intersection = line.intersect(j)

                #here "if" that checks if there is an intersection
                if intersection.size:
                    #evaluate tangent vector using built in method in bezier library
                    kassat=normalize(j.evaluate_hodograph(intersection[1,0]).reshape(1, 2))[0]

                    #reflect the direction of point over tangent vector
                    if np.dot(kassat, i.vector) < 0:
                        kassat = -kassat
                    i.vector = 2 * np.dot(i.vector, kassat) * kassat - i.vector

                    #change color
                    i.tail.set_color(tuple(np.random.rand(1, 3)[0]))


            #square bounce: the same as previous but for square
            x,y = i.next
            if abs(x)>5:
                kassat = np.array([0,1])
                if np.dot(kassat, i.vector) < 0:
                    kassat = -kassat
                i.vector = 2 * np.dot(i.vector, kassat) * kassat - i.vector
                i.tail.set_color(tuple(np.random.rand(1, 3)[0]))
            if abs(y)>5:
                kassat = np.array([1,0])
                if np.dot(kassat, i.vector) < 0:
                    kassat = -kassat
                i.vector = 2 * np.dot(i.vector, kassat) * kassat - i.vector
                i.tail.set_color(tuple(np.random.rand(1, 3)[0]))

            #change color of points=stars each frame
            i.point.set_color(tuple(np.random.rand(1, 3)[0]))

            #move points along their new direction=vector
            i.move()

        #map the data to animate it
        local=[i.point for i in list_points]+[i.tail for i in list_points]+b.list_curves_plt
        return local

    anim = FuncAnimation(fig, drawframe,frames=2000, interval=300/fps, blit=False)
    return anim

#--------------------------
#animation for bezier curve

# a=animate_bezier(20,1)
# plt.show()

#--------------------------
#animation for ellipse
def anime(speed_point, choice, num_point):
    plt.switch_backend('Agg') 
    #global x_input, y_input, speed_input, speed_point
    FPS = 30
    Npoints = 10

    #x_input, y_input = int(input('x - ')), int(input('y - '))
    print(type(speed_point))
    speed_point = int(speed_point)

    choice = choice

    #num_point = num_point
    if (choice == 1):
        animate_ellipse(10)
    else:
        animate_bezier(Npoints, FPS)
    buffer=BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph= base64.b64encode(image_png)
    graph=graph.decode('utf-8')
    buffer.close()
    time.sleep(0.05)
    return graph
    #plt.show()

    #o = animate_ellipse(10)

#o=animate_ellipse(10)
#plt.show()

##o=animate_ellipse(10,1)
##plt.show()
