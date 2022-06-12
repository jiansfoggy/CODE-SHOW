% Calculate cumulative probability for each pixel value
function sr = cumulative(pr, s_zero, L)
for i = 1:L
    for j = 1:i
        s_zero(i,2) = s_zero(i,2) + pr(j,2);
    end
end
sr=s_zero;
sr(:,2)=(sr(:,2)*(L-1));
end