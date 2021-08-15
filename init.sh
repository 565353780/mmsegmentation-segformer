mkdir pretrained
mkdir -p data/ade

cp ~/chLi/Download/DeepLearning/Model/SegFormer/trained_models/segformer.b5.640x640.ade.160k.pth ./
cp ~/chLi/Download/DeepLearning/Model/SegFormer/pretrained_models/mit_b5.pth ./pretrained/
cp ~/chLi/Download/DeepLearning/Dataset/ADE/ADEChallengeData2016.zip ./data/ade/

cd ./data/ade/
unzip ADEChallengeData2016.zip
rm ADEChallengeData2016.zip
cd ../../

