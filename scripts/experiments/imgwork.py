import torch
from diffusers import StableDiffusionPipeline
import os
os.environ["DISABLE_FLASH_ATTN"] = "1"
from PIL import Image
from transformers import CLIPImageProcessor
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")


champions = {
  "1": {
    "prompt": "Yasuo (left) and Riven (right) facing off on a bright urban street at midday. Passersby watch with awe and fear from the sidewalks. Tension builds as they prepare to clash. 9:16 vertical frame, cinematic light and wind effects."
  },
  "2": {
    "prompt": "Ahri and Zed stand in combat-ready poses on a busy city street in daylight. Pedestrians step back, watching nervously from storefronts and balconies. Magical energy shimmers between them in a 9:16 portrait shot."
  },
  "3": {
    "prompt": "Garen (left) and Darius (right) stare each other down in the middle of a sunny road. Cars stopped behind them, civilians recording with smartphones from the sidewalk. Bold sunlight and heat haze enhance the standoff."
  },
  "4": {
    "prompt": "Lux and Morgana stand with glowing magic ready on a clean city block. Office workers and shoppers gather around, stunned by the magical duel about to erupt. Portrait aspect, daylight with soft magical glow."
  },
  "5": {
    "prompt": "Jinx (left) wields her rocket launcher as Vi (right) raises her fists in the middle of a bustling shopping district street. People peek out from behind vending machines and store windows. High tension, 9:16 shot, vivid color."
  },
  "6": {
    "prompt": "Lee Sin and Sett face each other in fighting stances at a street corner, midday sun casting sharp shadows. Bystanders gather behind caution tape, filming with phones. Martial arts street duel energy, cinematic 9:16 layout."
  },
  "7": {
    "prompt": "Fiora and Camille circle each other on a tiled sidewalk in a high-end urban plaza. Elegant onlookers in suits and dresses stand at a distance. Steel and sunlight shimmer in the 9:16 vertical frame."
  },
  "8": {
    "prompt": "Katarina (left) and Akali (right) exchange deadly glances in the middle of a narrow alley street, lit by daylight from above. Civilians freeze in place, hiding near trash bins or watching from above. High tension urban ninja vibe."
  },
  "9": {
    "prompt": "Vayne and Draven lock eyes on a public square in a modern city, mid-day sun glowing. Spectators behind barricades whisper in anticipation. Vayne’s crossbow is raised, Draven spins his axes casually."
  },
  "10": {
    "prompt": "Shen and Master Yi prepare to duel on a broad pedestrian crossing. A traffic light blinks red while people across the street stop and stare. Tension, sunlight, and spiritual energy fill the air."
  },
  "11": {
    "prompt": "Tryndamere (left) roars in the middle of a commercial road while Renekton (right) snarls from across the street. Pedestrians rush away, others watch from inside a café. Dust and heat waves enhance the tension."
  },
  "12": {
    "prompt": "Irelia and Gwen stand ready on a wide street lined with trees and shops. Floating blades shimmer in sunlight. Curious children and adults peek from shop awnings, frozen in suspense. Magical urban atmosphere."
  },
  "13": {
    "prompt": "Aatrox and Kayn confront each other in broad daylight in a downtown avenue. Their dark energy distorts reality slightly. People flee in the distance, some frozen mid-step. Intense and apocalyptic urban showdown."
  },
  "14": {
    "prompt": "Swain and Malzahar begin chanting on opposite sides of a sunny city street, void and demon magic swirling. Cars halted mid-road, pedestrians watch in dread. Magical chaos brewing, captured in 9:16."
  },
  "15": {
    "prompt": "Ezreal (left) powers up his gauntlet while Lucian (right) aims his guns across a bright urban crosswalk. Street signs flicker from energy, and a crowd gathers behind barriers. Bright and heroic visual tone."
  },
  "16": {
    "prompt": "Rengar and Kha’Zix growl across a construction site avenue in broad daylight. Hard hats and workers stare from scaffoldings, some filming. A modern jungle setting with heavy industrial texture and predator tension."
  },
  "17": {
    "prompt": "Annie (left) with Tibbers summoned, and Sylas (right) gripping charged chains, face off in front of a modern mall entrance. Families stand frozen on the steps. Bright sun with glowing magical tension fills the air."
  },
  "18": {
    "prompt": "Orianna floats calmly at one end of a wide shopping street, Viktor crackling with hextech energy at the other. People watch from rooftop cafés and balconies. Daylight and tech energy blend in futuristic tension."
  },
  "19": {
    "prompt": "Thresh and Pyke stare each other down on a pier-side road in a coastal city. Green mist rises from sewer grates. Tourists and locals back away slowly as the atmosphere grows darker. Urban nautical gothic scene, 9:16 frame."
  },
  "20": {
    "prompt": "Nasus and Yorick face each other in a large plaza near a historic landmark, their presence towering over humans frozen in awe. Pigeons scatter, storm clouds begin to gather despite the daytime light. Ancient energy meets city life."
  }
}


# model_id = "Lykon/dreamshaper-8"
# model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
model_id = "black-forest-labs/FLUX.1-dev"

HF_TOKEN = os.getenv("HF_TOKEN", "hf_***MASKED***")

# === 모델 로드 ===
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")



os.makedirs("outputs", exist_ok=True)

for name, desc in champions.items():
    prompt = desc['prompt']  # ✅ 프롬프트 앞에 숫자 제거
    print(f"🎨 Generating {name}...")
    image = pipe(
    prompt=desc['prompt'],
    negative_prompt = "multiple people, extra bodies, extra limbs, twin faces, duplicate humans, crowd, more than one person",
    width=720,
    height=1024).images[0]
    image.save(f"outputs/{name}.png")


print("✅ All champion portraits saved in /outputs")
