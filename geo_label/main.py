import numpy as np
import cv2
import pyproj  # For latitude, longitude and UTM coordinate conversion
import PIL.Image

# Image pixel corners (assuming image size is W x H)
W, H = PIL.Image.open("nucleus/floor_1.png").size
print("Image size:", W, H)
pixel_coords = np.array([
    [0, 0],      # Top left
    [W, 0],      # Top right
    [0, H],      # Bottom left
    [W, H]       # Bottom right
], dtype=np.float32)

# NE and SW from user
lat_NE, lon_NE = 55.92332001571212, -3.1738768212979593
lat_SW, lon_SW = 55.92282257022002, -3.1745956532857647

# Interpolate the other two points
lat_NW = lat_NE
lon_NW = lon_SW

lat_SE = lat_SW
lon_SE = lon_NE

geo_coords_deg = np.array([
    [lat_NW, lon_NW],  # NW
    [lat_NE, lon_NE],  # NE
    [lat_SW, lon_SW],  # SW
    [lat_SE, lon_SE],  # SE
])

# Latitude and longitude → UTM conversion (we use the zone of lat1, lon1 by default)
proj_utm = pyproj.Proj(proj='utm', zone=33, ellps='WGS84', south=False)  # 你也可以用 pyproj.Transformer 自动确定
utm_coords = np.array([proj_utm(lon, lat) for lat, lon in geo_coords_deg], dtype=np.float32)

# Create perspective transformation (pixels → UTM coordinates)
H_matrix, _ = cv2.findHomography(pixel_coords, utm_coords)

# Query an obstacle pixel (x, y)
def pixel_to_geo(x, y):
    pixel = np.array([[x, y, 1]], dtype=np.float32).T
    utm = H_matrix @ pixel
    utm /= utm[2]  # Homogeneous normalization
    easting, northing = utm[0][0], utm[1][0]
    lon, lat = proj_utm(easting, northing, inverse=True)
    return lat, lon

def mark_pixel_on_map(image_path, x, y, output_path=None, show=True, radius=6, color=(0, 0, 255)):
    """
    Mark a pixel on the map image.

    Parameters:
    - image_path: str, map image path
    - x, y: int, pixel location
    - output_path: str, path to save the marked image (not saved if None)
    - show: bool, whether to use OpenCV to display the image window
    - radius: int, radius of the mark point
    - color: tuple, BGR color value, default red

    Return:
    - Marked image (numpy array)
    """

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Draw a point
    cv2.circle(img, (int(x), int(y)), radius=radius, color=color, thickness=-1)

    # show
    if show:
        cv2.imshow("Marked Map", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved Marked Images: {output_path}")

    return img

if __name__ == "__main__":
    # test
    x, y = 325, 555 # coordinate in pixels
    lat, lon = pixel_to_geo(x, y)
    print(f"Pixel ({x}, {y}) corresponds to Geo ({lat}, {lon})")

    # Mark this point on the map
    mark_pixel_on_map("nucleus/floor_1.png", x, y)

# Nucleus LG stairwell
# (350, 565) Geo (55.923021454852865, -3.17427175231112), (330, 585) Geo (55.92301092610767, -3.1742902930391566)
# (420, 565) Geo (55.92302142790645, -3.1742069340625343), (440, 585) Geo (55.92301088376152, -3.174188435819902)
# (420, 385) Geo (55.92311625586214, -3.1742067428110836), (440, 365) Geo (55.92312678459982, -3.174188202011103)
# (350, 385) Geo (55.92311628279836, -3.1742715612229624), (330, 365) Geo (55.92312682692639, -3.1742900595440013)

# Nucleus G stairwell
# (410, 580) Geo (55.92301090863321, -3.174221045188629), (410, 560) Geo (55.92302153551364, -3.1742210237571133)
# (325, 580) Geo (55.92301094092557, -3.1742987516650873), (305, 560) Geo (55.923021575395175, -3.1743170141360544)
# (325, 735) Geo (55.92292858261526, -3.174298917609189), (305, 755) Geo (55.92291796333051, -3.1743172228541)
# (445, 735) Geo (55.92292853699635, -3.174189214585424), (445, 755) Geo (55.922917910116, -3.1741892360311152)
# (445, 625) Geo (55.92298698483965, -3.1741890966446484), (465, 625) Geo (55.92298697722846, -3.174170812777022)

# Nucleus 1-G stairwell
# (400, 475) Geo (55.923068318751795, -3.174231922911987), (380, 495) Geo (55.92305776740818, -3.174250135524168)
# (475, 475) Geo (55.92306829037389, -3.1741637054250997), (495, 495) Geo (55.923057723893564, -3.1741455354069608)
# (475, 350) Geo (55.923134283538026, -3.1741635722639483), (495, 330) Geo (55.923144834872836, -3.174145359592135)
# (400, 350) Geo (55.92313431190845, -3.1742317898704404), (380, 330) Geo (55.92314487837232, -3.1742499599514304)

# Nucleus 2-1 stairwell
# (325, 580) Geo (55.923012912847625, -3.17430025206314)
# (445, 580) Geo (55.9230128674766, -3.1741911042535036)
# (445, 695) Geo (55.922952153773124, -3.174191226755363)
# (325, 695) Geo (55.922952199155134, -3.17430037438893)

# Nucleus Lift area
# (325, 485) Geo (55.92306141860183, -3.1742986499749017), (345, 465) Geo (55.92307361898947, -3.174281938426994)
# (200, 485) Geo (55.92306146598915, -3.1744129243400976), (180, 465) Geo (55.92307368121945, -3.174432016879645)
# (200, 545) Geo (55.92302958535738, -3.174412988467147), (180, 565) Geo (55.92302088671354, -3.1744321230471044)
# (245, 545) Geo (55.92302956830836, -3.1743718497330122), (265, 565) Geo (55.92302085467245, -3.1743548100179217)
# (245, 525) Geo (55.92304019518627, -3.1743718283451305), (265, 545) Geo (55.923031413574854, -3.1743547887615846)
# (325, 525) Geo (55.923040164843215, -3.1742986927902086), (345, 545) Geo (55.92303138337408, -3.174282023528208)

# gt around floor 1 stair well
# (325, 555) Geo (55.92302422452443, -3.174298724903236) tl
# (445, 555) Geo (55.92302417892292, -3.1741890216007107) tr
# (445, 755) Geo (55.922917910116, -3.1741892360311152) br
# (325, 755) Geo (55.92291795573684, -3.174298939023905) bl
