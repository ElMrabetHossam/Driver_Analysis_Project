import cv2
import numpy as np
import os

class LaneDetector:
    def __init__(self):
        # --- CONFIGURATION ---
        self.detected = False  
        
        # Mémoire
        self.recent_fit_left = []
        self.recent_fit_right = []
        self.best_fit_left = None
        self.best_fit_right = None
        
        # Buffer pour la stabilité
        self.buffer_size = 12 
        self.failure_count = 0 

    def get_perspective_transform(self, frame):
        h, w = frame.shape[:2]
        horizon_y = int(h * 0.50) 
        
        src = np.float32([
            [w * 0.46, horizon_y],    
            [w * 0.54, horizon_y],    
            [w * 0.98, h * 0.96],     
            [w * 0.02, h * 0.96]      
        ])
        
        dst = np.float32([
            [w * 0.20, 0], 
            [w * 0.80, 0], 
            [w * 0.80, h], 
            [w * 0.20, h]
        ])
        
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(frame, M, (w, h))
        return warped, Minv

    def robust_threshold(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        
        white_mask = cv2.inRange(cl, 200, 255)
        yellow_mask = cv2.inRange(lab, np.array([0, 0, 145]), np.array([255, 255, 255]))
        
        sobelx = cv2.Sobel(cl, cv2.CV_64F, 1, 0, ksize=9)
        abs_sobel = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sobel_mask = np.zeros_like(scaled_sobel)
        sobel_mask[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
        
        combined = np.zeros_like(white_mask)
        combined[((white_mask > 0) | (yellow_mask > 0)) | (sobel_mask == 1)] = 1
        return combined

    def search_around_poly(self, binary_warped):
        margin = 80 
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_fit = self.best_fit_left
        right_fit = self.best_fit_right
        
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & 
                          (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & 
                           (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
        
        return left_lane_inds, right_lane_inds, nonzerox, nonzeroy

    def sliding_window_search(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        midpoint = int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = int(binary_warped.shape[0]//nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        current_leftX = leftx_base
        current_rightX = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= current_leftX - margin) & (nonzerox < current_leftX + margin)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= current_rightX - margin) & (nonzerox < current_rightX + margin)).nonzero()[0]
            
            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)
            
            if len(good_left) > minpix: current_leftX = int(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix: current_rightX = int(np.mean(nonzerox[good_right]))

        return np.concatenate(left_lane_inds), np.concatenate(right_lane_inds), nonzerox, nonzeroy

    def sanity_check(self, left_fit, right_fit, h):
        if abs(left_fit[0] - right_fit[0]) > 0.005: 
            return False
        y_vals = [h-10, h/2]
        for y in y_vals:
            lx = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
            rx = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
            width = rx - lx
            if width < 300 or width > 1200: 
                return False
        return True

    def detect_lanes(self, frame):
        h, w = frame.shape[:2]
        warped, Minv = self.get_perspective_transform(frame)
        thresholded = self.robust_threshold(warped)
        
        if self.detected:
            left_inds, right_inds, nonzerox, nonzeroy = self.search_around_poly(thresholded)
        else:
            left_inds, right_inds, nonzerox, nonzeroy = self.sliding_window_search(thresholded)
        
        current_left_fit = None
        current_right_fit = None
        
        if len(left_inds) > 0 and len(right_inds) > 0:
            leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
            rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]
            try:
                current_left_fit = np.polyfit(lefty, leftx, 2)
                current_right_fit = np.polyfit(righty, rightx, 2)
            except: pass

        valid_detection = False
        if current_left_fit is not None and current_right_fit is not None:
            if self.sanity_check(current_left_fit, current_right_fit, h):
                valid_detection = True
                self.failure_count = 0 
            else: self.failure_count += 1
        else: self.failure_count += 1

        if valid_detection:
            self.detected = True
            self.best_fit_left = current_left_fit
            self.best_fit_right = current_right_fit
            self.recent_fit_left.append(current_left_fit)
            self.recent_fit_right.append(current_right_fit)
        else:
            if self.failure_count > 5:
                self.detected = False 
                self.recent_fit_left = [] 
                self.recent_fit_right = []
        
        if len(self.recent_fit_left) > self.buffer_size:
            self.recent_fit_left.pop(0)
            self.recent_fit_right.pop(0)
            
        if len(self.recent_fit_left) > 0:
            avg_left = np.mean(self.recent_fit_left, axis=0)
            avg_right = np.mean(self.recent_fit_right, axis=0)
        else: 
            # Si pas de détection, on renvoie None pour le polygone
            return frame, None 

        # --- DESSIN ---
        ploty = np.linspace(0, h-1, h)
        left_fitx = avg_left[0]*ploty**2 + avg_left[1]*ploty + avg_left[2]
        right_fitx = avg_right[0]*ploty**2 + avg_right[1]*ploty + avg_right[2]

        warp_zero = np.zeros_like(thresholded).astype(np.uint8)
        color_warp = cv2.merge((warp_zero, warp_zero, warp_zero))
        
        # Points pour le remplissage
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.concatenate((pts_left, pts_right), axis=1)

        # Points pour les lignes
        line_pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        line_pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])

        # Remplissage
        color_fill = (0, 255, 0) if self.failure_count < 10 else (0, 0, 255)
        cv2.fillPoly(color_warp, np.int_([pts]), color_fill)

        # Lignes
        cv2.polylines(color_warp, np.int_([line_pts_left]), False, (255, 0, 0), 20)
        cv2.polylines(color_warp, np.int_([line_pts_right]), False, (0, 0, 255), 20)

        # --- NOUVEAUTÉ : Calcul du Polygone sur l'image ORIGINALE ---
        # On utilise Minv pour projeter les points de la vue oiseau vers la vue normale
        lane_poly_original = cv2.perspectiveTransform(pts, Minv)

        # Projection finale
        newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
        result = cv2.addWeighted(frame, 1, newwarp, 0.5, 0)
        
        # On renvoie le résultat ET le polygone réel
        return result, lane_poly_original