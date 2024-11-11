class main:
    
    def __init__(self):
        pass
    

    def save_image_and_results(img, new_filename, detected_any, image_results, save_path, save_none_path):
        if detected_any:
            unique_save_path = os.path.join(save_path, new_filename)
            json_save_path = os.path.join(save_path, new_filename.replace(os.path.splitext(new_filename)[1], ".json"))
        else:
            unique_save_path = os.path.join(save_none_path, new_filename)
            json_save_path = os.path.join(save_none_path, new_filename.replace(os.path.splitext(new_filename)[1], ".json"))
            image_results["message"] = "No detections"

        # 保存图像
        cv2.imwrite(unique_save_path, img)
        print(f"Image saved: {unique_save_path}")

        # 保存 JSON 文件
        with open(json_save_path, 'w') as json_file:
            json.dump(image_results, json_file, indent=4)
            print(f"Results saved to: {json_save_path}")
