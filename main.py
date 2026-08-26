from dotenv import load_dotenv
import os
load_dotenv()



if __name__ == "__main__":
    if(os.environ.get("GOOGLE_API_KEY")):
        print("Key found")
    else:
        print("key not found")
        