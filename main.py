import numpy as np
import noise

class MapGenerator:
    def __init__(self,seed : int,width : int,height : int) -> None:
        self.seed = seed
        self.width = width
        self.height = height
        self.scale = 250
        x_idx = np.linspace(1, self.width, self.width)
        y_idx = np.linspace(1, self.height, self.height)
        self.__meshgrid = np.meshgrid(x_idx,y_idx)

        self.__maps = {
            "height_map": None,
            "land_map": None
        }

    def setScale(self,scale : float):
        self.scale = scale * 250

    def generateHeightMap(self):
        octaves = 4
        persistence = 0.6
        lacunarity = 2.0
        world_x,world_y = self.__meshgrid
        world_z = np.vectorize(noise.pnoise2)(
                                world_x/(self.scale),
                                world_y/(self.scale),
                                octaves=octaves,
                                persistence=persistence,
                                lacunarity=lacunarity,
                                repeatx=1024,
                                repeaty=1024,
                                base=self.seed)

        def filter(x : float) -> float:
            if x <= 0.1:
                return -0.4
            elif x <= 0.15:
                return 7.5*x - 1.15
            elif x <= 0.2:
                return 0.5 * x - 0.1
            elif x <= 0.3:
                return 0
            elif x <= 0.6:
                return (0.6*x) - 0.18
            else:
                return (x/15) + 0.113

        world_z -= np.min(world_z)
        world_z /= np.max(world_z)
        world_z = np.vectorize(filter)(world_z)
        world_z *= 6000

        self.__maps["height_map"] = world_z
        return world_z

    def show_height_map(self, map_type : str):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator

        map = self.__maps[map_type]
        world_x,world_y = self.__meshgrid

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(18,18))
        surf = ax.plot_surface(world_x,world_y,map,cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        scaler = max(abs(np.min(map)),abs(np.max(map)))#type: ignore
        ax.set_zlim(-1.01 * scaler, 1.01 * scaler)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

if __name__ == "__main__":
    map_generator = MapGenerator(1410,640,640)
    map_generator.generateHeightMap()
    map_generator.show_height_map("height_map")
