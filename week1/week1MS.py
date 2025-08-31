import cv2
import os


input_filename = '/home/jason/projects/openvc/week1/shutterstock130285649--250.jpg'


home_directory = os.path.expanduser('~')
output_filename = os.path.join(home_directory, 'Desktop', 'numbers_copy.jpg')


if not os.path.exists(input_filename):
    print(f"Error: The file '{input_filename}' was not found.")
    print("Please make sure the file path is correct.")
else:
    try:
        print(f"Loading the image: {input_filename}")
        image = cv2.imread(input_filename)
        if image is None:
            print(f"Error: Could not decode the image. The file '{input_filename}' might be corrupted.")
        else:
            print("Image loaded successfully.")
            window_title = 'Numbers Image (Press any key to close)'
            cv2.imshow(window_title, image)
            print(f"Displaying the image in a window titled '{window_title}'.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Image window closed.")


            success = cv2.imwrite(output_filename, image)

            if success:
                output_path = os.path.abspath(output_filename)
                print(f"Successfully wrote a copy of the image to: {output_path}")
            else:
                print(f"Error: Could not write the image to the file '{output_filename}'.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")