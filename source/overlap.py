import matplotlib
import numpy as np


def overlab(x,y,list_of_annotations,ax,verbose=False,**kwargs):
    
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    
    def doOverlap(ret1,ret2):
        l1 = Point(ret1[0,0],ret1[1,1])
        r1 = Point(ret1[1,0],ret1[0,1])
        l2 = Point(ret2[0,0],ret2[1,1])
        r2 = Point(ret2[1,0],ret2[0,1])

        # If one rectangle is on left side of other
        if l1.x >= r2.x or l2.x >= r1.x:
            return False

        # If one rectangle is above other
        if(r1.y >= l2.y or r2.y >= l1.y):
            return False

        return True

    annotations_coord=[]
    for i, dot in enumerate(y):
        x_coords=x[i]
        y_coords=y[i]
        annotation=ax.annotate(str(list_of_annotations[i]),
                                xy=(x[i],y[i]),
                                 xytext=(x_coords,y_coords),
                                    **kwargs)

        ax.figure.canvas.draw()
        bbox=matplotlib.text.Text.get_window_extent(annotation)
        bbox_data = ax.transData.inverted().transform(bbox)
        factor=0.2*(bbox_data[0,0]-bbox_data[1,0])
        annotations_coord.append(bbox_data)
        ##BUILD THE SPIRAL##
        theta=np.radians(np.linspace(1,360*200,500))
        r=np.linspace(0,max(max(zip(x,y))),len(theta))
        x_2 = r*np.cos(theta)+x_coords#move the spiral onto the data point
        y_2 = r*np.sin(theta)+y_coords
        n=0
        keep_cycling=True
        while keep_cycling:
            keep_cycling=False
            if verbose==True:
                print('start checking box %s'% i)
            for ind, box in enumerate (annotations_coord[0:-1]):
                if verbose:
                    print('checking %s and %s' % (i,ind))
                if doOverlap(box,bbox_data):
                    if verbose:
                        print('%s and %s overlap' % (i,ind))
                    annotation.set_x(x_2[n])
                    annotation.set_y(y_2[n])
                    n+=1
                    ax.figure.canvas.draw()
                    bbox=matplotlib.text.Text.get_window_extent(annotation)
                    bbox_data = ax.transData.inverted().transform(bbox)
                    annotations_coord.pop()
                    annotations_coord.append(bbox_data)
                    if verbose:
                        print('new coords (x=%i,y=%i)'%(x_coords,y_coords))
                        print('new bbox data',bbox_data)
                        print('annotation coordinates',box)
                        print('restart iteration')
                    keep_cycling=True
                    break






def plot_enrichment_analisys(df,p_value):
    cmap = plt.get_cmap("tab10")
    colors=cmap(np.arange(len(df.source.value_counts())))
    ylim=max(-np.log10(df.p_value.tolist()))
    for i, (s,v) in enumerate(zip(df.source.value_counts().index,df.source.value_counts())):
        x=sorted(np.random.uniform(low=0.0, high=1, size=v))
        y=-np.log10(df[df.source==s].p_value.tolist())
        fig,ax=plt.subplots(figsize=(15,10))
        scat=ax.scatter(x,y,color=colors[i],alpha=0.4,
                    s=np.clip(df[df.source==s].term_size.tolist(),200,2000))
        ax.set_xlabel(s,fontsize=20)
        ax.set_ylabel('$-\log_{10}(p\ value)$',fontsize=20)
        ax.spines['bottom'].set_color(colors[i])
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(color=colors[i], linestyle='-', linewidth=2,alpha=0.05)
        ax.set_xticks([])
        ax.tick_params(direction='out',which= 'major' ,length=8, width=2, left=False)
        ax.set_ylim(0,max(-np.log10(df.p_value.tolist())))
        
        
        
        ## ADD LABELS
        bbox_props = dict(boxstyle="round", fc="w", ec="k", lw=1)
        handles=[]
        lables=[]
        linea_y=np.linspace(ax.get_ylim()[1],0,
                            sum(map(lambda x: -np.log10(x)>=p_value,df[df.source==s].p_value.tolist())))
        h=0
        for ind, pv in enumerate(y):
            factory=np.random.uniform(-2,5,1)
            factorx=np.random.uniform(-0.1,0.1,1)
            if pv > p_value:
                handles.append(str(h))
                lables.append(df[df.source==s].name.tolist()[ind])
                an=ax.annotate(str(df[df.source==s].name.tolist()[ind]),
                                 xy=(x[ind],y[ind]),
                                 xytext=(ax.get_xlim()[1]+0.2, linea_y[ind]),
                                 #arrowprops=dict(arrowstyle="-",shrinkB=10),
                               bbox=bbox_props,
                                    fontsize=12)
                ax.annotate('',
                                 xy=(x[ind],y[ind]),
                                 xytext=(ax.get_xlim()[1]+0.2, linea_y[ind]),
                                 arrowprops=dict(arrowstyle="-",shrinkB=10,linewidth=0.2),
                               #bbox=bbox_props,
                                    fontsize=12)
                
                h+=1
        
        
        plt.show()