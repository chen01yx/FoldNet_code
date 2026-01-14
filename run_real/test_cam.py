import argparse
from garmentds.real_galbot.gsocket_utils import GalbotClient
import cv2
import numpy as np

def test_cam():
    client = GalbotClient()
    
    rgbd = client.get_rgbd()
    while True:
        rgbd = client.get_rgbd()
        color, depth = rgbd['color'], rgbd['depth']
        cv2.imshow('color', color[..., ::-1])
        cv2.imshow('depth', cv2.cvtColor((np.clip(depth, 0., 2.) / 2. * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # test_server()
    test_cam()