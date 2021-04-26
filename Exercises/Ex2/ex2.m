M = randi(100, [5 6]);                      %# input matrix
% M = 1:25;
% k=5;
% M = reshape(M, [fix(size(M,2)/k),k]);

M
zigZagArray = zigZag(M)
restoreM = zigZagInverse(zigZagArray, size(M))

isequal(M, restoreM)


function zigZagArray = zigZag(mat)
	ind = reshape(1:numel(mat), size(mat));   %# indices of elements
	ind = fliplr(spdiags(fliplr(ind)));       %# get the anti-diagonals
	ind(:,1:2:end) = flipud(ind(:,1:2:end));  %# reverse order of odd columns
	ind(ind==0) = [];                         %# keep non-zero indices
	zigZagArray = mat(ind);
end

function mat = zigZagInverse(zigZagArray, imSize)
	M = 1:length(zigZagArray);
	M = reshape(M, imSize);
	indices = zigZag(M);
	mat = zeros(imSize);
	for i = 1:length(indices)
		index = indices(i);
		f = fix((index-1)/imSize(2))+1;
		r = rem(index-1,imSize(2))+1;
		mat(f,r) = zigZagArray(i);
	end
	mat = reshape(mat.', imSize);
end