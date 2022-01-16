# install requirements.txt
python3 -m pip install -r requirements.txt 

git clone https://github.com/CompVis/taming-transformers
git clone https://github.com/openai/CLIP

# Download ImageNet VQGAN
curl -L -o vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' 
curl -L -o vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' 

# Download wikiART VQGAN
# curl -L -o wikiart_16384.yaml -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml' 
# curl -L -o wikiart_16384.ckpt -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt' 

# Download sketchy dataset
wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1z4--ToTXYb0-2cLuUWPYM5m7ST7Ob3Ck' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1z4--ToTXYb0-2cLuUWPYM5m7ST7Ob3Ck" -O rendered_256x256.7z && rm -rf /tmp/cookies.txt
7z x rendered_256x256.7z