from PIL import ImageTk, Image, ImageDraw
import PIL

basewidth = 64
img = Image.open('test.jpg')
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save('test.jpg') 
