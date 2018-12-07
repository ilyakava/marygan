for i=1:size(X,4)
    imwrite(uint8(X(:,:,:,i)), sprintf('/Users/artsyinc/Documents/MATH630/research/data/svhn/img/%05d.jpg',i));
end