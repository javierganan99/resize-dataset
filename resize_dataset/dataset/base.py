from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class ResizableDataset(Dataset, ABC):
    """
    Abstract base class for resizable datasets.

    This class defines a framework for datasets that support resizing images
    and their corresponding annotations. It must be subclassed to implement
    specific dataset formats.

    Methods:
        scale(img, anns, scale_factor, resize_image_method): Scales the given image
            and annotations by a scale factor.
        show(image, anns): Displays the image along with its annotations.
        __getitem__(index): Retrieves the image and annotations at the specified index.
        __len__(): Returns the total number of items in the dataset.
    """

    @abstractmethod
    def __init__(self, images_path, annotations_path, cfg):
        """
        Initializes the object with the specified image and annotation paths,
        along with configuration settings.

        Args:
            images_path (str): The path to the directory/file containing the images.
            annotations_path (str): The path to the directory/file containing the annotations.
            cfg (ConfigDict): Configuration settings for the object, including parameters for
                processing images and annotations.

        Returns:
            None
        """

    @abstractmethod
    def scale(self, img, anns, scale_factor, resize_image_method):
        """
        Scales an image and its associated annotations by a specified factor.

        This function adjusts the size of the input image and modifies the annotations accordingly
        to maintain their relevance after scaling. The image can be resized using different methods
        specified by the user.

        Args:
            img (np.ndarray): The image to be scaled, represented as a NumPy array.
            anns (list): A list of annotations associated with the image, which will also be scaled.
            scale_factor (float): The factor by which to scale the image and annotations.
            resize_image_method (str): The method used for resizing the image.

        Returns:
            tuple: A tuple containing the scaled image (np.ndarray)
                and the modified annotations (list).
        """

    @abstractmethod
    def reshape(self, img, anns, shape, resize_image_method):
        """
        Reshape an image and its associated annotations by a specified factor.

        This function adjusts the size of the input image and modifies the annotations accordingly
        to maintain their relevance after resizing. The image can be resized using different methods
        specified by the user.

        Args:
            img (np.ndarray): The image to be scaled, represented as a NumPy array.
            anns (list): A list of annotations associated with the image, which will also be scaled.
            shape (tuple): The new shape of the image and annotations.
            resize_image_method (str): The method used for resizing the image.

        Returns:
            tuple: A tuple containing the scaled image (np.ndarray)
                and the modified annotations (list).
        """

    @abstractmethod
    def show(self, image, anns):
        """
        Displays an image along with its associated annotations.

        This function is designed to visualize an image along with any given annotations,
        allowing users to easily understand the content of the image and the context
        provided by the annotations.

        Args:
            image (np.ndarray): The image to be displayed.
            anns (List[AnnotationType]): A list of annotations to overlay on the image.
                Each annotation should include necessary details depending on the task.
        """

    @abstractmethod
    def save(self, index, image, anns):
        """
        Saves an image along with its associated annotations.

        This function is designed to save an image along with any given annotations.

        Args:
            image (np.ndarray): The image to be displayed.
            anns (List[AnnotationType]): A list of annotations to be saved.
        """

    @abstractmethod
    def __getitem__(self, index):
        """
        Retrieve an item from a data structure using the specified index.

        This method allows access to elements in the object's internal storage
        by providing an integer index. It raises an IndexError if the index
        is out of range.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the processed image (np.ndarray) and
                a list of 1 element containing the annotations (List[AnnotationType]).
        """

    @abstractmethod
    def __len__(self):
        """
        Returns the number of images of the dataset.

        Returns:
            int: Number of images of the dataset.
        """

    @abstractmethod
    def close(self):
        """
        Perform the needed operations when closing the dataset,
        such as saving information, closing files, etc.
        """
