import os
import json
import time
import requests
from database import db, ImageRecord, init_db
from similarity import ImageSimilarityMatcher
from flask import Flask

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///image_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

REFERENCE_FOLDER = 'reference_images'
PICSUM_URL = "https://picsum.photos/seed/{seed}/800/600"
HEADERS = {"User-Agent": "Mozilla/5.0"}

DIVERSE_IMAGES = [
    ("mountain", "mountain.jpg", "Mountain landscape"),
    ("mountains2", "mountains.jpg", "Mountain range"),
    ("mountain3", "mountain_snow.jpg", "Snowy mountain peak"),
    ("valley", "valley.jpg", "Valley view"),
    ("lake", "lake.jpg", "Lake and water"),
    ("river", "river.jpg", "River or stream"),
    ("waterfall", "waterfall.jpg", "Waterfall"),
    ("beach", "beach.jpg", "Beach and shore"),
    ("beach2", "beach_sunset.jpg", "Beach at sunset"),
    ("ocean", "ocean.jpg", "Ocean view"),
    ("ocean2", "ocean_waves.jpg", "Ocean waves"),
    ("forest", "forest.jpg", "Forest scene"),
    ("forest2", "forest_path.jpg", "Path through forest"),
    ("tree", "tree.jpg", "Tree in landscape"),
    ("trees", "trees.jpg", "Trees and greenery"),
    ("woods", "woods.jpg", "Woods or forest"),
    ("meadow", "meadow.jpg", "Meadow or field"),
    ("hills", "hills.jpg", "Rolling hills"),
    ("cliff", "cliff.jpg", "Cliff or rocks"),
    ("canyon", "canyon.jpg", "Canyon landscape"),
    ("desert", "desert.jpg", "Desert scene"),
    ("desert2", "desert_dunes.jpg", "Desert dunes"),
    ("sky", "sky.jpg", "Sky and clouds"),
    ("clouds", "clouds.jpg", "Cloudy sky"),
    ("sunset", "sunset.jpg", "Sunset sky"),
    ("sunrise", "sunrise.jpg", "Sunrise"),
    ("night", "night.jpg", "Night scene"),
    ("stars", "stars.jpg", "Night sky or stars"),
    ("rain", "rain.jpg", "Rain or wet weather"),
    ("snow", "snow.jpg", "Snow scene"),
    ("ice", "ice.jpg", "Ice or frost"),
    ("grass", "grass.jpg", "Grass or lawn"),
    ("field", "field.jpg", "Open field"),
    ("garden", "garden.jpg", "Garden"),
    ("park", "park.jpg", "Park or green space"),
    ("flower", "flower.jpg", "Single flower"),
    ("flowers", "flowers.jpg", "Flowers"),
    ("rose", "rose.jpg", "Rose flower"),
    ("tulip", "tulip.jpg", "Tulips"),
    ("sunflower", "sunflower.jpg", "Sunflowers"),
    ("leaf", "leaf.jpg", "Leaf or foliage"),
    ("leaves", "leaves.jpg", "Leaves"),
    ("city", "city.jpg", "City skyline"),
    ("city2", "city_night.jpg", "City at night"),
    ("building", "building.jpg", "Building or structure"),
    ("buildings", "buildings.jpg", "Buildings"),
    ("skyscraper", "skyscraper.jpg", "Skyscraper"),
    ("bridge", "bridge.jpg", "Bridge"),
    ("road", "road.jpg", "Road or path"),
    ("street", "street.jpg", "Street scene"),
    ("highway", "highway.jpg", "Highway or road"),
    ("house", "house.jpg", "House or home"),
    ("church", "church.jpg", "Church building"),
    ("tower", "tower.jpg", "Tower"),
    ("monument", "monument.jpg", "Monument"),
    ("fountain", "fountain.jpg", "Fountain"),
    ("stairs", "stairs.jpg", "Stairs or steps"),
    ("window", "window.jpg", "Window"),
    ("door", "door.jpg", "Door"),
    ("wall", "wall.jpg", "Wall or surface"),
    ("roof", "roof.jpg", "Roof or building top"),
    ("office", "office.jpg", "Office interior"),
    ("kitchen", "kitchen.jpg", "Kitchen"),
    ("room", "room.jpg", "Room interior"),
    ("interior", "interior.jpg", "Interior space"),
    ("car", "car.jpg", "Car or vehicle"),
    ("cars2", "cars.jpg", "Cars on street"),
    ("car2", "car_red.jpg", "Red car"),
    ("bike", "bike.jpg", "Bicycle"),
    ("motorcycle", "motorcycle.jpg", "Motorcycle"),
    ("truck", "truck.jpg", "Truck"),
    ("bus", "bus.jpg", "Bus"),
    ("boat", "boat.jpg", "Boat on water"),
    ("ship", "ship.jpg", "Ship or boat"),
    ("plane", "plane.jpg", "Airplane"),
    ("train", "train.jpg", "Train"),
    ("person", "person.jpg", "Person"),
    ("people", "people.jpg", "People"),
    ("person2", "person_portrait.jpg", "Portrait photo"),
    ("person3", "person_walking.jpg", "Person walking"),
    ("crowd", "crowd.jpg", "Crowd of people"),
    ("child", "child.jpg", "Child"),
    ("hands", "hands.jpg", "Hands"),
    ("face", "face.jpg", "Face or portrait"),
    ("dog", "dog.jpg", "Dog"),
    ("dog2", "dog_portrait.jpg", "Dog portrait"),
    ("cat", "cat.jpg", "Cat"),
    ("cat2", "cat_sleeping.jpg", "Cat resting"),
    ("animal", "animal.jpg", "Animal"),
    ("bird", "bird.jpg", "Bird"),
    ("birds", "birds.jpg", "Birds"),
    ("horse", "horse.jpg", "Horse"),
    ("fish", "fish.jpg", "Fish"),
    ("butterfly", "butterfly.jpg", "Butterfly"),
    ("lion", "lion.jpg", "Lion"),
    ("elephant", "elephant.jpg", "Elephant"),
    ("deer", "deer.jpg", "Deer"),
    ("rabbit", "rabbit.jpg", "Rabbit"),
    ("cow", "cow.jpg", "Cow"),
    ("sheep", "sheep.jpg", "Sheep"),
    ("food", "food.jpg", "Food dish"),
    ("food2", "food_plate.jpg", "Food on plate"),
    ("coffee", "coffee.jpg", "Coffee"),
    ("coffee2", "coffee_cup.jpg", "Coffee cup"),
    ("fruit", "fruit.jpg", "Fruit"),
    ("fruits", "fruits.jpg", "Bowl of fruit"),
    ("apple", "apple.jpg", "Apple or fruit"),
    ("vegetables", "vegetables.jpg", "Vegetables"),
    ("salad", "salad.jpg", "Salad"),
    ("bread", "bread.jpg", "Bread"),
    ("pizza", "pizza.jpg", "Pizza"),
    ("cake", "cake.jpg", "Cake or dessert"),
    ("wine", "wine.jpg", "Wine glass"),
    ("bottle", "bottle.jpg", "Bottle"),
    ("bottles2", "bottles.jpg", "Bottles"),
    ("glass", "glass.jpg", "Glass or drink"),
    ("cup", "cup.jpg", "Cup or mug"),
    ("meal", "meal.jpg", "Meal"),
    ("breakfast", "breakfast.jpg", "Breakfast"),
    ("book", "book.jpg", "Book"),
    ("books", "books.jpg", "Stack of books"),
    ("laptop", "laptop.jpg", "Laptop computer"),
    ("phone", "phone.jpg", "Phone or mobile"),
    ("camera", "camera.jpg", "Camera"),
    ("watch", "watch.jpg", "Watch"),
    ("keyboard", "keyboard.jpg", "Keyboard"),
    ("lamp", "lamp.jpg", "Lamp"),
    ("chair", "chair.jpg", "Chair"),
    ("table", "table.jpg", "Table"),
    ("sofa", "sofa.jpg", "Sofa or couch"),
    ("bed", "bed.jpg", "Bed or bedroom"),
    ("mirror", "mirror.jpg", "Mirror"),
    ("painting", "painting.jpg", "Painting or art"),
    ("sculpture", "sculpture.jpg", "Sculpture"),
    ("vase", "vase.jpg", "Vase"),
    ("candle", "candle.jpg", "Candle"),
    ("plant", "plant.jpg", "Potted plant"),
    ("pot", "pot.jpg", "Pot or container"),
    ("bag", "bag.jpg", "Bag"),
    ("shoes", "shoes.jpg", "Shoes"),
    ("hat", "hat.jpg", "Hat"),
    ("umbrella", "umbrella.jpg", "Umbrella"),
    ("glasses", "glasses.jpg", "Glasses"),
    ("pen", "pen.jpg", "Pen"),
    ("notebook", "notebook.jpg", "Notebook"),
    ("guitar", "guitar.jpg", "Guitar"),
    ("piano", "piano.jpg", "Piano"),
    ("ball", "ball.jpg", "Ball"),
    ("toy", "toy.jpg", "Toy"),
    ("keys", "keys.jpg", "Keys"),
    ("tool", "tool.jpg", "Tool"),
    ("box", "box.jpg", "Box"),
    ("texture", "texture.jpg", "Texture or surface"),
    ("pattern", "pattern.jpg", "Pattern"),
    ("abstract", "abstract.jpg", "Abstract image"),
    ("color", "color.jpg", "Colorful image"),
    ("light", "light.jpg", "Light"),
    ("shadow", "shadow.jpg", "Shadow"),
    ("reflection", "reflection.jpg", "Reflection"),
    ("blur", "blur.jpg", "Blurred image"),
    ("minimal", "minimal.jpg", "Minimal style"),
    ("vintage", "vintage.jpg", "Vintage style"),
    ("retro", "retro.jpg", "Retro style"),
    ("modern", "modern.jpg", "Modern style"),
    ("dark", "dark.jpg", "Dark scene"),
    ("bright", "bright.jpg", "Bright scene"),
    ("red", "red.jpg", "Red tones"),
    ("blue", "blue.jpg", "Blue tones"),
    ("green", "green.jpg", "Green tones"),
    ("yellow", "yellow.jpg", "Yellow tones"),
    ("orange", "orange.jpg", "Orange tones"),
    ("purple", "purple.jpg", "Purple tones"),
    ("white", "white.jpg", "White or light"),
    ("black", "black.jpg", "Black or dark"),
    ("market", "market.jpg", "Market scene"),
    ("restaurant", "restaurant.jpg", "Restaurant"),
    ("cafe", "cafe.jpg", "Cafe"),
    ("library", "library.jpg", "Library"),
    ("museum", "museum.jpg", "Museum"),
    ("stadium", "stadium.jpg", "Stadium"),
    ("airport", "airport.jpg", "Airport"),
    ("station", "station.jpg", "Train or bus station"),
    ("port", "port.jpg", "Port or harbor"),
    ("farm", "farm.jpg", "Farm"),
    ("factory", "factory.jpg", "Factory"),
    ("construction", "construction.jpg", "Construction site"),
    ("rocks", "rocks.jpg", "Rocks"),
    ("sand", "sand.jpg", "Sand"),
    ("water", "water.jpg", "Water"),
    ("fire", "fire.jpg", "Fire or flames"),
    ("smoke", "smoke.jpg", "Smoke"),
    ("fog", "fog.jpg", "Fog or mist"),
    ("mist", "mist.jpg", "Mist"),
    ("autumn", "autumn.jpg", "Autumn foliage"),
    ("winter", "winter.jpg", "Winter scene"),
    ("spring", "spring.jpg", "Spring scene"),
    ("summer", "summer.jpg", "Summer scene"),
]


def download_image(seed, filepath):
    url = PICSUM_URL.format(seed=seed)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


def add_to_db(filepath, filename, description, matcher):
    existing = ImageRecord.query.filter_by(filename=filename).first()
    if existing:
        print("    Already in database, skipping.")
        return True
    try:
        try:
            from caption import get_caption
            desc = get_caption(filepath)
            if desc:
                description = desc
        except Exception:
            pass
        embedding = matcher.get_image_embedding(filepath)
        embedding_json = json.dumps(embedding.tolist())
        record = ImageRecord(
            filename=filename,
            filepath=filepath,
            description=description or "Image",
            embedding=embedding_json,
        )
        db.session.add(record)
        db.session.commit()
        print("    ok id", record.id)
        return True
    except Exception as e:
        db.session.rollback()
        print("    err", e)
        return False


def main():
    os.makedirs(REFERENCE_FOLDER, exist_ok=True)
    print("Loading CLIP...")
    with app.app_context():
        init_db()
        matcher = ImageSimilarityMatcher()
        added = 0
        failed = 0
        for seed, filename, description in DIVERSE_IMAGES:
            filepath = os.path.join(REFERENCE_FOLDER, filename)
            print(f"\n{filename} — {description}")
            if not os.path.exists(filepath):
                if not download_image(seed, filepath):
                    failed += 1
                    continue
                time.sleep(0.3)
            if add_to_db(filepath, filename, description, matcher):
                added += 1
            else:
                failed += 1
        total = ImageRecord.query.count()
        print("Done. added", added, "failed", failed, "total", total)


if __name__ == "__main__":
    main()
