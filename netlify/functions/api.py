from mangum import Mangum
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from app import app

handler = Mangum(app)