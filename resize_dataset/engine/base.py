from tqdm import tqdm
from resize_dataset.utils import (
    DEFAULT_CFG,
    colorstr,
    ConfigDict,
    TQDM_BAR_FORMAT_GREEN,
)
from resize_dataset.dataset import DATASET_REGISTRY


class ResizeDataset:
    """
    Resizes and processes datasets based on specified tasks.

    This class is used to resize and visualize datasets according to the configured task.
    It loads the dataset from a specified format and applies transformations based on
    the parameters passed in.

    Args:
        task (str): The task to perform on the dataset (e.g., 'scale').
        **kwargs: Additional configuration parameters for the dataset loading.

    Attributes:
        task (str): The name of the task to perform on the dataset.
        cfg (ConfigDict): The configuration dictionary combining default and provided settings.
        dataset (Dataset): The loaded dataset based on the specified configuration.

    Methods:
        scale_dataset(): Scales the dataset images and optionally displays them with annotations.

    Properties:
        task_map: A dictionary mapping task names to their corresponding methods.
    """

    def __init__(self, task, **kwargs):
        """
        Initializes an instance of the class with the specified task and configuration.

        This constructor loads the dataset according to the provided configuration and
        calls the task indicated by the 'task' parameter. The dataset is obtained from
        the DATASET_REGISTRY using the dataset format and task specified in the config.

        Args:
            task (str): The name of the task to be executed.
            **kwargs: Additional keyword arguments that override the default configuration
                      settings in DEFAULT_CFG.

        Returns:
            None: This constructor does not return any value.
        """
        self.task = task
        self.cfg = ConfigDict({**DEFAULT_CFG, **kwargs})
        # Load the dataset
        self.dataset = DATASET_REGISTRY.get(self.cfg.dataset_format).get(
            self.cfg.dataset_task
        )(self.cfg.images_path, self.cfg.labels_path, self.cfg)
        # Call the indicated task
        self.task_map[self.task]()

    def scale_dataset(self):
        """
        Scales the dataset and displays images with annotations if configured to do so.

        This method iterates over the dataset using a DataLoader with a batch size of 1.
        For each image and its corresponding annotations, it checks the configuration
        setting and displays the image if the 'show' option is enabled.

        Args:
            self: The instance of the class that holds the dataset and configuration.

        Returns:
            None: This function does not return any value.
        """
        i = 0
        for _, _ in tqdm(
            self.dataset,
            desc=colorstr("📏 Scaling dataset"),
            total=len(self.dataset),
            bar_format=TQDM_BAR_FORMAT_GREEN,
            colour="green",
        ):
            pass
        self.dataset.close()

    @property
    def task_map(self):
        """
        Defines the tasks to be performed by the task map.

        This function creates a mapping of task names to their respective methods,
        enabling structured execution of specified tasks within the class.

        Returns:
            dict: A dictionary mapping task names (str) to corresponding methods (callable).
        """
        return {
            "scale": self.scale_dataset,
        }
