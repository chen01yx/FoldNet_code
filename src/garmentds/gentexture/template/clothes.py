import numpy as np

import cv2
from PIL import Image

from garmentds.gentexture.utils.clients import *

class Default_Client():
    def __init__(self):
        pass
    def get_response(self, *args):
        return "A piece of cloth fabric with (minor wrinkles) AND ((small number of logos:1.5))."\
               "The fabric should be multicolored and vibrant."

class Base:
    def __init__(
        self, flipped=False, client=None,
        use_same_front_back=True, 
        use_symmetric_texture=True,
    ):
        self.use_same_front_back = use_same_front_back
        self.use_symmetric_texture = use_symmetric_texture
        self.flipped = flipped

        if client == "dmiapi":
            self.client = DMIAPI_Client()
        elif client == "openai":
            self.client = OpenAI_Client()
        else:
            self.client = Default_Client()

        self.client_prompt_template = \
            "Design a prompt for stable diffusion. The prompt is used to generate " \
            "texture image for a piece of flat cloth fabric and should not exceed 70 words. " \
            "The texture image generated should be {hue} for the base color, and decorated " \
            "with moderate number of logos or patterns of text and graphics. The fabric is " \
            "used to make {object_name}. The response you give should not directly " \
            "mention the name: {object_name}, and you should just describe attributes " \
            "of the fabric that is used to make {object_name}. Emphasize the flatness "\
            "of the fabric and the decorations on the fabric at the beginning of your response. "\
            "When you emphasize some words, just place them between parenthesis, the more "\
            "important the word is, the more layers of parenthesis should be used. Don't "\
            "just place adjective in the parenthsis, include the noun or verb that is related "\
            "to the adjective together in the parenthsis. "            

    def get_client_prompt_template(self):
        return self.client_prompt_template

    def __generate_single_prompt__(self, object_name:str):
        """
            Generate single prompt for diffusion.
        """
        p = np.random.rand()
        if p < 0.2:
            hue = "multi-colored but not garish"
        elif p < 0.7:
            hue = "limited in color scheme but not monochrome"
        else:
            hue = "single hue"

        # like roughness, weight, " f"stretch and so on. 
        client_prompt = self.client_prompt_template.format(hue=hue, object_name=object_name)

        message = [
            {
                "role": "system",
                "content": """
                    You are a helpful AI assistant.
                """
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": client_prompt}],
            },
        ]

        while True:
            diffusion_prompt = self.client.get_response(message)
            if diffusion_prompt is None:
                print("[ ERROR ] Client Failed. Retrying...")
                continue
            # filter out too long prompts for diffusion
            num_tokens = len(diffusion_prompt.split())
            if num_tokens <= 77:
                print(f"[ INFO ] Diffusion prompt({num_tokens}/77 tokens): {diffusion_prompt}")
                break
            else:
                print(f"[ INFO ] Prompt is too long({num_tokens}/77 tokens). Retrying...")
        return diffusion_prompt

    def generate_prompts(self):
        pass

    def union_images(self, images):
        """
            Union images for each parts of garment.
        """
        pass

    def generate_keys(self):
        """
            Give keypoint names on the garment.
        """
        pass

    def set_flipped(self, flipped):
        self.flipped = flipped

class TShirtSim(Base):
    def __init__(
        self, client=None,
        use_same_front_back=True, 
        use_symmetric_texture=False,
    ):
        super().__init__(
            use_same_front_back=use_same_front_back, 
            use_symmetric_texture=use_symmetric_texture,
            client=client)

    def generate_prompts(self):
        diffusion_prompt = self.__generate_single_prompt__("tshirt")
        
        return [[diffusion_prompt]] * 4
    
    def union_images(self, images):
        np_images = [np.array(image) for image in images]
        np_front, np_back, np_sleeve, np_collar = np_images[:4]
        np_icons = np_images[4:]
        resized_front = cv2.resize(np_front, (512, 512))
        resized_back = cv2.resize(np_back, (512, 512))
        resized_sleeve = cv2.resize(np_sleeve, (512, 512))
        resized_collar = cv2.resize(np_collar, (512, 512))
        resized_icons = [cv2.resize(np_icon, (64, 64)) for np_icon in np_icons]
        ## TODO: add icons
        # front_blend = cv2.addWeighted(resized_front[360:424, 224:288, :], 0.5, resized_icon, 0.5, 1.0, 0.0)
        # resized_front[360:424, 224:288, :] = front_blend
        # Image.fromarray(np.uint8(front_blend)).save("blend.png")
        union = np.zeros((1024, 1024, 3))
        union[:512, :512, :] = resized_front
        if self.use_same_front_back:
            union[512:, :512, :] = resized_front
        else:
            union[512:, :512, :] = resized_back

        union[:512, 512:, :] = resized_collar
        union[512:, 512:, :] = resized_sleeve
        return union

    def generate_keys(self):
        if not self.flipped:
            return ["r_collar_o", "r_shoulder", "r_sleeve_top", 
                    "r_sleeve_bottom", "r_armpit", "r_corner", 
                    "l_corner", "l_armpit", "l_sleeve_bottom", 
                    "l_sleeve_top", "l_shoulder", "l_collar_o", 
                    "neck_b_o", "neck_f_o"]
        else:
            return ["l_collar_o", "l_shoulder", "l_sleeve_top", 
                    "l_sleeve_bottom", "l_armpit", "l_corner", 
                    "r_corner", "r_armpit", "r_sleeve_bottom", 
                    "r_sleeve_top", "r_shoulder", "r_collar_o", 
                    "neck_b_o", "neck_f_o"]


class TShirtSPSim(Base):
    def __init__(
        self, client=None,
        use_same_front_back=True, 
        use_symmetric_texture=False,
    ):
        super().__init__(
            use_same_front_back=use_same_front_back, 
            use_symmetric_texture=use_symmetric_texture,
            client=client)

    def generate_prompts(self):
        diffusion_prompt = self.__generate_single_prompt__("tshirt")
        
        if self.use_same_front_back:
            return [[diffusion_prompt]]
        else:
            return [[diffusion_prompt],[diffusion_prompt]]
    
    def union_images(self, images):
        np_images = [np.array(image) for image in images]

        if len(np_images) >= 2: 
            np_front, np_back = np_images[:2]
        else:
            np_front = np_images[0]
            np_back = np_images[0].copy()

        resized_front = cv2.resize(np_front, (512, 512))
        resized_back = cv2.resize(np_back, (512, 512))
        if self.use_symmetric_texture:
            resized_front[:, :256] = resized_front[:, -1:255:-1]
            resized_back[:, :256] = resized_back[:, -1:255:-1]

        union = np.zeros((1024, 1024, 3))
        union[:512, :512, :] = resized_front
        if self.use_same_front_back:
            union[512:, :512, :] = resized_front
        else:
            union[512:, :512, :] = resized_back
        return union
    
    def generate_keys(self):
        if not self.flipped:
            return ["r_collar_o", "r_shoulder", "r_sleeve_top", 
                    "r_sleeve_bottom", "r_armpit", "r_corner", 
                    "l_corner", "l_armpit", "l_sleeve_bottom", 
                    "l_sleeve_top", "l_shoulder", "l_collar_o", 
                    "neck_b_o", "neck_f_o"]
        else:
            return ["l_collar_o", "l_shoulder", "l_sleeve_top", 
                    "l_sleeve_bottom", "l_armpit", "l_corner", 
                    "r_corner", "r_armpit", "r_sleeve_bottom", 
                    "r_sleeve_top", "r_shoulder", "r_collar_o", 
                    "neck_b_o", "neck_f_o"] 

class TrousersSim(Base):
    def __init__(
        self, client=None,
        use_same_front_back=True,
        use_symmetric_texture=False, 
    ):
        super().__init__(
            use_same_front_back=use_same_front_back,
            use_symmetric_texture=use_symmetric_texture,
            client=client)

    def generate_prompts(self):
        diffusion_prompt = self.__generate_single_prompt__("trousers")

        if self.use_same_front_back:
            return [[diffusion_prompt]]
        else:
            return [[diffusion_prompt],[diffusion_prompt]]
    
    def union_images(self, images):
        np_images = [np.array(image) for image in images]

        if len(np_images) >= 2: 
            np_front, np_back = np_images[:2]
        else:
            np_front = np_images[0]
            np_back = np_images[0].copy()

        resized_front = cv2.resize(np_front, (512, 512))
        resized_back = cv2.resize(np_back, (512, 512))
        if self.use_symmetric_texture:
            resized_front[:, :256] = resized_front[:, -1:255:-1]
            resized_back[:, :256] = resized_back[:, -1:255:-1]

        union = np.zeros((1024, 1024, 3))
        union[:512, :512, :] = resized_front
        if self.use_same_front_back:
            union[512:, :512, :] = resized_front
        else:
            union[512:, :512, :] = resized_back
        return union
    
    def generate_keys(self):
        if not self.flipped:
            return ["r_corner", "r_leg_o", "r_leg_i", 
                    "crotch", "l_leg_i", "l_leg_o", 
                    "l_corner", "top_ctr_f"]
        else:
            return ["l_corner", "l_leg_o", "l_leg_i", 
                    "crotch", "r_leg_i", "r_leg_o", 
                    "r_corner", "top_ctr_f"]

class VestCloseSPSim(Base):
    def __init__(
        self, client=None,
        use_same_front_back=True,
        use_symmetric_texture=False, 
    ):
        super().__init__(
            use_same_front_back=use_same_front_back,
            use_symmetric_texture=use_symmetric_texture, 
            client=client)

    def generate_prompts(self):
        vest_prompt = self.__generate_single_prompt__("vest")

        if self.use_same_front_back:
            return [[vest_prompt]]
        else:
            return [[vest_prompt],[vest_prompt]]
    
    def union_images(self, images):
        np_images = [np.array(image) for image in images]

        if len(np_images) >= 2: 
            np_front, np_back = np_images[:2]
        else:
            np_front = np_images[0]
            np_back = np_images[0].copy()

        resized_front = cv2.resize(np_front, (512, 512))
        resized_back = cv2.resize(np_back, (512, 512))
        if self.use_symmetric_texture:
            resized_front[:, :256] = resized_front[:, -1:255:-1]
            resized_back[:, :256] = resized_back[:, -1:255:-1]

        union = np.zeros((1024, 1024, 3))
        union[:512, :512, :] = resized_front
        if self.use_same_front_back:
            union[512:, :512, :] = resized_front
        else:
            union[512:, :512, :] = resized_back
        return union
    
    def generate_keys(self):
        if not self.flipped:
            return ["r_collar", "r_shoulder", "r_armpit", "r_corner",
                    "l_corner", "l_armpit", "l_shoulder", "l_collar",
                    "neck_b_o", "neck_f_o"]
        else:
            return ["l_collar", "l_shoulder", "l_armpit", "l_corner",
                    "r_corner", "r_armpit", "r_shoulder", "r_collar",
                    "neck_b_o", "neck_f_o"]


class HoodedCloseSim(Base):
    def __init__(
        self, client=None,
        use_same_front_back=True, 
        use_symmetric_texture=False,
    ):
        super().__init__(
            use_same_front_back=use_same_front_back,
            use_symmetric_texture=use_symmetric_texture, 
            client=client)

    def generate_prompts(self):
        hoodie_prompt = self.__generate_single_prompt__("hoodie")

        if self.use_same_front_back:
            return [[hoodie_prompt]]
        else:
            return [[hoodie_prompt], [hoodie_prompt]]
    
    def union_images(self, images):
        np_images = [np.array(image) for image in images]

        if len(np_images) > 1: 
            np_front, np_back = np_images[:2]
        else:
            np_front = np_images[0]
            np_back = np_front.copy()
        np_hood = np_back.copy()

        resized_front = cv2.resize(np_front, (512, 512))
        resized_back = cv2.resize(np_back, (512, 512))
        resized_hood = cv2.resize(np_hood, (512, 512))
        if self.use_symmetric_texture:
            resized_front[:, :256] = resized_front[:, -1:255:-1]
            resized_back[:, :256] = resized_back[:, -1:255:-1]

        union = np.zeros((1024, 1024, 3))
        union[512:, :512, :] = resized_front
        if self.use_same_front_back:
            union[:512, :512, :] = resized_front
        else:
            union[:512, :512, :] = resized_back
        union[:512, 512:, :] = resized_hood
        return union
    
    def generate_keys(self):
        if not self.flipped:
            return ["r_collar", "r_shoulder", "r_sleeve_top", 
                    "r_sleeve_bottom", "r_armpit", "r_corner", 
                    "l_corner", "l_armpit", "l_sleeve_bottom", 
                    "l_sleeve_top", "l_shoulder", "l_collar",
                    "hood_top", "neck_f"]
        else:
            return ["l_collar", "l_shoulder", "l_sleeve_top", 
                    "l_sleeve_bottom", "l_armpit", "l_corner", 
                    "r_corner", "r_armpit", "r_sleeve_bottom", 
                    "r_sleeve_top", "r_shoulder", "r_collar",
                    "hood_top", "neck_f"]
  

if __name__ == "__main__":
    toursers = TrousersSim(use_same_front_back=True)
    print(toursers.generate_prompts())