clc
clear variables
close all

H = imread('haze.bmp');
% imshowFigure(H)

I = rgb2gray(H);
% imshowFigure(I)

negativeImage = negativeIm(I);
% imshowFigure(negativeImage)

padded = whitePadIm(I,[10,10]);
% imshowFigure(padded)

flipped = flipHorizontal(I);
% imshowFigure(flipped)

structMat = splitMat2Struct(I,[5 5]);
% newMat = structMat2Mat(structMat);

flippedStruct = applayFunc2StructMat(structMat, @flipHorizontal);
newFlippedMat = structMat2Mat(flippedStruct);
imshowFigure(newFlippedMat)

negativeStruct = applayFunc2StructMat(structMat, @negativeIm);
newNegativedMat = structMat2Mat(negativeStruct);
imshowFigure(newNegativedMat)

paddedStruct = applayFunc2StructMat(structMat, @(im) whitePadIm(im, [1,1]));
newPaddeddMat = structMat2Mat(paddedStruct);
imshowFigure(newPaddeddMat)

function imshowFigure(im)
	figure
	imshow(im)
end

function newtructMat = applayFunc2StructMat(structMat, func)
	structMatSize = size(structMat);
	for i = 1:structMatSize(1)
		for j = 1:structMatSize(2)
			structMat(i,j).submat = func(structMat(i,j).submat);
			structMat(i,j).origin = [1 + (i-1)*size(structMat(i,j).submat,1), 1 + (j-1)*size(structMat(i,j).submat,2)];
		end
	end

	newtructMat = structMat;
end

function structMat = splitMat2Struct(mat, blockSizeArray)
	matSize = size(mat);
	reshapedMat = permute(reshape(mat, blockSizeArray(1), matSize(1) / blockSizeArray(1), blockSizeArray(2), matSize(2) / blockSizeArray(2)), [2, 4, 1, 3]);
	structMat(matSize(1)/blockSizeArray(1),matSize(2)/blockSizeArray(2)).submat = zeros(size(reshapedMat(1,1,:,:)), class(size(reshapedMat(1,1,:,:))));
	structMat(matSize(1)/blockSizeArray(1),matSize(2)/blockSizeArray(2)).origin = [0,0];
	for i = 1:matSize(1)/blockSizeArray(1)
		for j = 1:matSize(2)/blockSizeArray(2)
			bla = reshapedMat(i,j,:,:);
			structMat(i,j).submat = reshape(bla, blockSizeArray(1), blockSizeArray(2));
			structMat(i,j).origin = [1 + (i-1)*blockSizeArray(1), 1 + (j-1)*blockSizeArray(2)];
		end
	end
end


function mat = structMat2Mat(structMat)
	structMatSize = size(structMat);
	blockSize = [size(structMat(1,1).submat,1), size(structMat(1,1).submat,2)];
	mat = zeros([structMatSize(1)*blockSize(1), structMatSize(2)*blockSize(2)], class(structMat(1,1).submat));
	for i = 1:structMatSize(1)
		for j = 1:structMatSize(2)
			mat(structMat(i,j).origin(1):structMat(i,j).origin(1)+blockSize(1)-1,...
				structMat(i,j).origin(2):structMat(i,j).origin(2)+blockSize(2)-1) = structMat(i,j).submat;
		end
	end
end

function negative = negativeIm(im)
	negative = 255 - im;
end

function flipped = flipHorizontal(mat)
	flipped = flip(mat,2);
end

function padded = whitePadIm(im, paddingValArray)
	padded = ones(size(im,1)+paddingValArray(1),size(im,2)+paddingValArray(2),'like',im)*255;
	padded(paddingValArray(1)/2:end-(paddingValArray(1)/2+1),paddingValArray(2)/2:end-(paddingValArray(2)/2+1)) = im;
end