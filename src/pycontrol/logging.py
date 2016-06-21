import logging

logger = logging.getLogger('pycontrol')
logging.basicConfig(format='%(name)s-%(levelname)s: %(message)s')
logger.setLevel(logging.INFO)
