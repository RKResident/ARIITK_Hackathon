# This is the unfixed version of the generator.py code
# Hybrid Marker Generator — IT IS DELIBERATELY POORLY TUNED
# Your job is to fix the sigma values
#
# Usage  : python generator.py <far_img> <near_img> <aruco_id> <output>
# Example: python generator.py far_target_decoy.png aruco_markers/aruco_id_37.png 37 hybrid_marker.png

import cv2
import numpy as np
import sys


def create_hybrid_marker(far_path, near_path, aruco_id, output_path,
                          sigma_low=5, sigma_high=4):
    """
    Creates a Hybrid Marker by combining a far (decoy) image and a near (ArUco) image
    using spatial frequency filtering.

    
    Parameters:
        far_path   : Path to the decoy image (H helipad symbol)
        near_path  : Path to the ArUco marker image
        aruco_id   : The ArUco ID being encoded (for reporting only)
        output_path: Output filename for the hybrid marker
        sigma_low  : Gaussian sigma for low-pass filter on decoy  ← FIX THIS
        sigma_high : Gaussian sigma for high-pass filter on ArUco ← FIX THIS
    """

    print(f"\n--- Hybrid Marker Generator ---")
    print(f"Far image  (decoy)  : {far_path}")
    print(f"Near image (ArUco)  : {near_path}")
    print(f"ArUco ID            : {aruco_id}")
    print(f"sigma_low           : {sigma_low}  ← fix this")
    print(f"sigma_high          : {sigma_high}  ← fix this")
    print(f"")

    # Create your hybrid marker here by applying appropriate filters to the far and near images and save them to output_path.     
    print(f"Hybrid marker saved → {output_path}")
    print(f"")
    print(f"Next steps:")
    print(f"  python simulator.py {output_path} 30   ← should show H symbol, no ArUco")
    print(f"  python simulator.py {output_path} 2    ← should show ArUco clearly")
    print(f"  python visual_check.py {output_path} {far_path}   ← check ghosting score")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage  : python generator.py <far_img> <near_img> <aruco_id> <output>")
        print("Example: python generator.py far_target_decoy.png aruco_markers/aruco_id_37.png 37 hybrid_marker.png")
        sys.exit(1)

    create_hybrid_marker(
        far_path    = sys.argv[1],
        near_path   = sys.argv[2],
        aruco_id    = int(sys.argv[3]),
        output_path = sys.argv[4]
    )