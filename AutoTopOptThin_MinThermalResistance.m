function AutoTopOptThin_MinThermalResistance(nelx,nely,volfrac,rmin,maxiter)
% ------------------------------------------------------------
% 2D thermal topology optimization
% Case:
%   - uniform heat generation over the whole domain
%   - one point sink on the left boundary
%   - bi-material design (high-k / low-k)
%   - minimize thermal resistance by reducing the temperature field
%
% Improved version:
%   - density filter
%   - Heaviside projection
%   - penal continuation
%   - beta continuation
%
% Usage:
%   AutoTopOptThin_MinThermalResistance;
%   AutoTopOptThin_MinThermalResistance(240,140,0.15,1.8,300);
% ------------------------------------------------------------

if nargin < 1, nelx    = 240; end
if nargin < 2, nely    = 140; end
if nargin < 3, volfrac = 0.15; end
if nargin < 4, rmin    = 1.8;  end
if nargin < 5, maxiter = 300;  end

% ---- material ----
kHigh = 1e3;
kLow  = 1.0;

% ---- continuation parameters ----
penal    = 1.0;
penalMax = 6.0;

beta     = 1.0;
betaMax  = 32.0;
eta      = 0.50;   % projection threshold

% ---- FE matrices ----
KE = 1/6 * [ 4 -1 -2 -1;
            -1  4 -1 -2;
            -2 -1  4 -1;
            -1 -2 -1  4];

FE = [1;1;1;1]/4;  % uniform heat generation per element

% ---- mesh ----
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofMat = [reshape(nodenrs(1:end-1,1:end-1),nelx*nely,1), ...
           reshape(nodenrs(1:end-1,2:end  ),nelx*nely,1), ...
           reshape(nodenrs(2:end  ,2:end  ),nelx*nely,1), ...
           reshape(nodenrs(2:end  ,1:end-1),nelx*nely,1)];

iK = reshape(kron(edofMat,ones(4,1))',16*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,4))',16*nelx*nely,1);

ndof = (nelx+1)*(nely+1);

% ---- internal heat generation ----
iF = reshape(edofMat',4*nelx*nely,1);
sF = repmat(FE,nelx*nely,1);
F = sparse(iF,1,sF,ndof,1);

% ---- right boundary plate sink ----
midy = round((nely+1)/2);
sink_half_width = max(1, round(0.10*(nely+1)));
sink_nodes = nodenrs(max(1,midy-sink_half_width):min(nely+1,midy+sink_half_width),end);
fixeddofs = unique(sink_nodes(:));
alldofs   = (1:ndof)';
freedofs  = setdiff(alldofs,fixeddofs);

% ---- filter ----
iH = [];
jH = [];
sH = [];
for i1 = 1:nelx
    for j1 = 1:nely
        e1 = (i1-1)*nely + j1;
        for i2 = max(i1-floor(rmin),1):min(i1+floor(rmin),nelx)
            for j2 = max(j1-floor(rmin),1):min(j1+floor(rmin),nely)
                e2 = (i2-1)*nely + j2;
                fac = rmin - sqrt((i1-i2)^2 + (j1-j2)^2);
                if fac > 0
                    iH = [iH; e1];
                    jH = [jH; e2];
                    sH = [sH; fac];
                end
            end
        end
    end
end
H  = sparse(iH,jH,sH);
Hs = sum(H,2);

% ---- initial design ----
x = volfrac * ones(nely,nelx);

loop   = 0;
change = 1.0;

figure(1); clf;

while change > 1e-3 && loop < maxiter
    loop = loop + 1;

    % ---- density filter ----
    xTilde = reshape((H*x(:))./Hs,nely,nelx);

    % ---- Heaviside projection ----
    denom = tanh(beta*eta) + tanh(beta*(1-eta));
    xPhys = (tanh(beta*eta) + tanh(beta*(xTilde-eta))) / denom;

    % ---- FE analysis ----
    kval = kLow + xPhys(:).^penal * (kHigh-kLow);
    sK = reshape(KE(:) * kval',16*nelx*nely,1);
    K  = sparse(iK,jK,sK);
    K  = (K+K')/2;

    T = zeros(ndof,1);
    Kff = K(freedofs,freedofs);
    Kff = (Kff + Kff')/2;
    Kff = Kff + 1e-9 * speye(length(freedofs));
    T(freedofs) = Kff \ F(freedofs);
    T(fixeddofs) = 0;

    % ---- objective: minimize average temperature under fixed generation ----
    c = full(sum(T));

    % ---- adjoint for gradient ----
    lambda = zeros(ndof,1);
    lambda(freedofs) = Kff \ ones(numel(freedofs),1);

    Te = T(edofMat);
    Le = lambda(edofMat);
    dc_phys_e = sum((Le*KE).*Te,2);
    dc_phys = -penal*(kHigh-kLow) * xPhys.^(penal-1) .* reshape(dc_phys_e,nely,nelx);

    % ---- projection derivative ----
    dxPhys_dxTilde = beta * (1 - tanh(beta*(xTilde-eta)).^2) / denom;

    % ---- chain rule back to design variables x ----
    tmp_dc = (dc_phys(:).*dxPhys_dxTilde(:))./Hs;
    dc = reshape(H' * tmp_dc, nely, nelx);
    onevec = ones(nelx*nely,1);
    tmp_dv = (onevec .* dxPhys_dxTilde(:))./Hs;
    dv = reshape(H' * tmp_dv, nely, nelx);

    % ---- OC update ----
    l1 = 0; l2 = 1e12; move = 0.15;
    while (l2-l1)/(l2+l1+eps) > 1e-4
        lmid = 0.5*(l1+l2);
        B = -dc ./ dv / lmid;
        B(B < 1e-30) = 1e-30;
        xnew = max(0, max(x-move, min(1, min(x+move, x.*sqrt(B)))));
        xTildeNew = reshape((H*xnew(:))./Hs,nely,nelx);
        xPhysNew  = (tanh(beta*eta) + tanh(beta*(xTildeNew-eta))) / denom;
        if mean(xPhysNew(:)) > volfrac
            l1 = lmid;
        else
            l2 = lmid;
        end
    end

    change = max(abs(xnew(:)-x(:)));
    x = xnew;

    % ---- continuation ----
    if mod(loop,40)==0
        penal = min(penal+1, penalMax);
    end
    if mod(loop,50)==0
        beta = min(beta*2, betaMax);
    end

    % ---- print ----
    fprintf('It.:%4d  Obj.:%11.4e  Vol.:%6.3f  ch.:%7.4f  Tmax:%9.4e  penal:%4.1f  beta:%4.1f\n',... 
        loop,c,mean(xPhys(:)),change,max(T),penal,beta);

    % ---- draw every 10 iters ----
    if mod(loop,10)==1 || loop==maxiter || change<1e-3
        imagesc(flipud(xPhys));
        axis equal off;
        caxis([0 1]);
        colormap(gray);
        title(sprintf('iter=%d, obj=%.3e, vol=%.3f, penal=%.1f, beta=%.1f', ...
              loop,c,mean(xPhys(:)),penal,beta));
        drawnow;
    end
end

outdir = fullfile(pwd, 'results');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

fprintf('\nDone.\n');
fprintf('White = high-k material, Black = low-k material.\n');

figure(2); clf;
imagesc(flipud(xPhys));
axis equal off;
colormap(gray);
title('Optimized design for minimum thermal resistance');
saveas(gcf, fullfile(outdir, 'min_resistance_design.png'));

% ---- CV transient response on optimized design ----
% left boundary pulse, right boundary point sink

tau_q = 0.10;
t_end = 0.20;
dt = 5e-4;
 x_cv = linspace(0.0, 1.0, nelx+1)';
 y_cv = linspace(0.0, 1.0, nely+1)';
 dx = x_cv(2) - x_cv(1);
 dy = y_cv(2) - y_cv(1);
 t_grid = 0:dt:t_end;
 nt = numel(t_grid);

kPhys = kLow + xPhys.^penal * (kHigh - kLow);
k_node = zeros(nelx+1,nely+1);
w_node = zeros(nelx+1,nely+1);
for ie = 1:nelx
    for je = 1:nely
        ke = kPhys(je,ie);
        k_node(ie,je) = k_node(ie,je) + ke; w_node(ie,je) = w_node(ie,je) + 1;
        k_node(ie+1,je) = k_node(ie+1,je) + ke; w_node(ie+1,je) = w_node(ie+1,je) + 1;
        k_node(ie+1,je+1) = k_node(ie+1,je+1) + ke; w_node(ie+1,je+1) = w_node(ie+1,je+1) + 1;
        k_node(ie,je+1) = k_node(ie,je+1) + ke; w_node(ie,je+1) = w_node(ie,je+1) + 1;
    end
end
k_node = k_node ./ w_node;

nsub = max(1, ceil(dt / min([0.20 / (2.0 * (1.0/(dx*dx) + 1.0/(dy*dy))), 0.35 * tau_q, 0.35 * min(dx,dy) / sqrt(1.0/max(tau_q,1e-30))])));
dt_sub = dt / nsub;

T_cv = zeros(nelx+1,nely+1);
qx = zeros(nelx+1,nely+1);
qy = zeros(nelx+1,nely+1);
T_time = zeros(nelx+1,nely+1,nt);

for it = 1:nt
    for js = 1:nsub
        t_now = (it-1)*dt + (js-1)*dt_sub;
        qL = exp(-0.5*((t_now-0.030)/0.003)^2);
        qR = 0.0;
        qL_row = qL * ones(1,nely+1);

        gx = zeros(nelx+1,nely+1);
        gy = zeros(nelx+1,nely+1);
        gx(2:end-1,:) = (T_cv(3:end,:) - T_cv(1:end-2,:)) / (2*dx);
        gx(1,:) = (T_cv(2,:) - T_cv(1,:)) / dx;
        gx(end,:) = (T_cv(end,:) - T_cv(end-1,:)) / dx;
        gy(:,2:end-1) = (T_cv(:,3:end) - T_cv(:,1:end-2)) / (2*dy);
        gy(:,1) = (T_cv(:,2) - T_cv(:,1)) / dy;
        gy(:,end) = (T_cv(:,end) - T_cv(:,end-1)) / dy;

        dqx_dt = (-k_node .* gx - qx) / tau_q;
        dqy_dt = (-k_node .* gy - qy) / tau_q;
        qx_new = qx + dt_sub * dqx_dt;
        qy_new = qy + dt_sub * dqy_dt;
        qx_new(1,:) = qL_row;
        qx_new(end,:) = qR;
        qy_new(:,1) = 0.0;
        qy_new(:,end) = 0.0;

        dqx = zeros(nelx+1,nely+1);
        dqy = zeros(nelx+1,nely+1);
        dqx(2:end-1,2:end-1) = (qx_new(3:end,2:end-1) - qx_new(1:end-2,2:end-1)) / (2*dx);
        dqy(2:end-1,2:end-1) = (qy_new(2:end-1,3:end) - qy_new(2:end-1,1:end-2)) / (2*dy);
        dqx(1,2:end-1) = (qx_new(2,2:end-1) - qL_row(2:end-1)) / dx;
        dqy(1,2:end-1) = (qy_new(1,3:end) - qy_new(1,1:end-2)) / (2*dy);
        dqx(end,2:end-1) = (qR - qx_new(end-1,2:end-1)) / dx;
        dqy(end,2:end-1) = (qy_new(end,3:end) - qy_new(end,1:end-2)) / (2*dy);
        dqy(2:end-1,1) = (qy_new(2:end-1,2) - 0.0) / dy;
        dqx(2:end-1,1) = (qx_new(3:end,1) - qx_new(1:end-2,1)) / (2*dx);
        dqy(2:end-1,end) = (0.0 - qy_new(2:end-1,end-1)) / dy;
        dqx(2:end-1,end) = (qx_new(3:end,end) - qx_new(1:end-2,end)) / (2*dx);
        dqx(1,1) = (qx_new(2,1) - qL_row(1)) / dx;
        dqy(1,1) = (qy_new(1,2) - 0.0) / dy;
        dqx(1,end) = (qx_new(2,end) - qL_row(end)) / dx;
        dqy(1,end) = (0.0 - qy_new(1,end-1)) / dy;
        dqx(end,1) = (qR - qx_new(end-1,1)) / dx;
        dqy(end,1) = (qy_new(end,2) - 0.0) / dy;
        dqx(end,end) = (qR - qx_new(end-1,end)) / dx;
        dqy(end,end) = (0.0 - qy_new(end,end-1)) / dy;

        T_cv = T_cv - dt_sub * (dqx + dqy) ./ k_node;
        qx = qx_new;
        qy = qy_new;
    end
    T_time(:,:,it) = T_cv;
end

snapshot_times = [0.04, 0.10, 0.20];
figure(3); clf;
for k = 1:numel(snapshot_times)
    [~, idx] = min(abs(t_grid - snapshot_times(k)));
    subplot(1,numel(snapshot_times),k);
    imagesc(x_cv, y_cv, T_time(:,:,idx)');
    axis equal tight;
    colorbar;
    xlabel('x^*');
    ylabel('y^*');
    title(sprintf('CV T at t=%.3f', t_grid(idx)));
end
sgtitle('2D CV temperature snapshots');
saveas(gcf, fullfile(outdir, 'minres_cv_snapshots.png'));

figure(4); clf;
imagesc(t_grid, x_cv, squeeze(T_time(:,round((nely+1)/2),:)));
axis xy;
colorbar;
xlabel('t^*');
ylabel('x^*');
title('CV spacetime at mid-y slice');
saveas(gcf, fullfile(outdir, 'minres_cv_spacetime.png'));

% ---- Thermomass transient response on optimized design ----
tau_tm = 0.20;
eps_tm = 0.02;

T_tm = zeros(nelx+1,nely+1);
qx_tm = zeros(nelx+1,nely+1);
qy_tm = zeros(nelx+1,nely+1);
T_time_tm = zeros(nelx+1,nely+1,nt);

for it = 1:nt
    for js = 1:nsub
        t_now = (it-1)*dt + (js-1)*dt_sub;
        qL = exp(-0.5*((t_now-0.030)/0.003)^2);
        qR = 0.0;
        qL_row = qL * ones(1,nely+1);

        gx = zeros(nelx+1,nely+1);
        gy = zeros(nelx+1,nely+1);
        gx(2:end-1,:) = (T_tm(3:end,:) - T_tm(1:end-2,:)) / (2*dx);
        gx(1,:) = (T_tm(2,:) - T_tm(1,:)) / dx;
        gx(end,:) = (T_tm(end,:) - T_tm(end-1,:)) / dx;
        gy(:,2:end-1) = (T_tm(:,3:end) - T_tm(:,1:end-2)) / (2*dy);
        gy(:,1) = (T_tm(:,2) - T_tm(:,1)) / dy;
        gy(:,end) = (T_tm(:,end) - T_tm(:,end-1)) / dy;

        Theta = 1.0 + eps_tm * T_tm;
        Theta_s = max(Theta, 1.0e-12);
        ux = qx_tm ./ Theta_s;
        uy = qy_tm ./ Theta_s;

        qx_up = zeros(nelx+1,nely+1);
        for j = 1:nely+1
            for i = 2:nelx
                if ux(i,j) >= 0
                    qx_up(i,j) = (qx_tm(i,j) - qx_tm(i-1,j)) / dx;
                else
                    qx_up(i,j) = (qx_tm(i+1,j) - qx_tm(i,j)) / dx;
                end
            end
            qx_up(1,j) = (qx_tm(2,j) - qL_row(j)) / dx;
            qx_up(end,j) = (qR - qx_tm(end-1,j)) / dx;
        end

        qy_uy = zeros(nelx+1,nely+1);
        for i = 1:nelx+1
            for j = 2:nely
                if uy(i,j) >= 0
                    qy_uy(i,j) = (qy_tm(i,j) - qy_tm(i,j-1)) / dy;
                else
                    qy_uy(i,j) = (qy_tm(i,j+1) - qy_tm(i,j)) / dy;
                end
            end
            qy_uy(i,1) = (qy_tm(i,2) - 0.0) / dy;
            qy_uy(i,end) = (0.0 - qy_tm(i,end-1)) / dy;
        end

        div_theta = zeros(nelx+1,nely+1);
        div_theta(2:end-1,2:end-1) = (qx_tm(3:end,2:end-1) - qx_tm(1:end-2,2:end-1)) / (2*dx) + (qy_tm(2:end-1,3:end) - qy_tm(2:end-1,1:end-2)) / (2*dy);
        div_theta(1,2:end-1) = (qx_tm(2,2:end-1) - qL_row(2:end-1)) / dx + (qy_tm(1,3:end) - qy_tm(1,1:end-2)) / (2*dy);
        div_theta(end,2:end-1) = (qR - qx_tm(end-1,2:end-1)) / dx + (qy_tm(end,3:end) - qy_tm(end,1:end-2)) / (2*dy);
        div_theta(2:end-1,1) = (qx_tm(3:end,1) - qx_tm(1:end-2,1)) / (2*dx) + (qy_tm(2:end-1,2) - 0.0) / dy;
        div_theta(2:end-1,end) = (qx_tm(3:end,end) - qx_tm(1:end-2,end)) / (2*dx) + (0.0 - qy_tm(2:end-1,end-1)) / dy;
        div_theta(1,1) = (qx_tm(2,1) - qL_row(1)) / dx + (qy_tm(1,2) - 0.0) / dy;
        div_theta(1,end) = (qx_tm(2,end) - qL_row(end)) / dx + (0.0 - qy_tm(1,end-1)) / dy;
        div_theta(end,1) = (qR - qx_tm(end-1,1)) / dx + (qy_tm(end,2) - 0.0) / dy;
        div_theta(end,end) = (qR - qx_tm(end-1,end)) / dx + (0.0 - qy_tm(end,end-1)) / dy;

        Theta_t = -eps_tm * div_theta;
        Theta_x = eps_tm * gx;
        Theta_y = eps_tm * gy;

        rqx = ux .* Theta_t - gx / tau_tm - eps_tm * ux .* (qx_up - ux .* Theta_x);
        rqy = uy .* Theta_t - gy / tau_tm - eps_tm * uy .* (qy_uy - uy .* Theta_y);
        qx_star = (qx_tm + dt_sub * rqx) / (1.0 + dt_sub / tau_tm);
        qy_star = (qy_tm + dt_sub * rqy) / (1.0 + dt_sub / tau_tm);
        dqx_dt = (qx_star - qx_tm) / dt_sub;
        dqy_dt = (qy_star - qy_tm) / dt_sub;

        qx_new = qx_tm + dt_sub * dqx_dt;
        qy_new = qy_tm + dt_sub * dqy_dt;
        qx_new(1,:) = qL_row;
        qx_new(end,:) = qR;
        qy_new(:,1) = 0.0;
        qy_new(:,end) = 0.0;

        dqx = zeros(nelx+1,nely+1);
        dqy = zeros(nelx+1,nely+1);
        dqx(2:end-1,2:end-1) = (qx_new(3:end,2:end-1) - qx_new(1:end-2,2:end-1)) / (2*dx);
        dqy(2:end-1,2:end-1) = (qy_new(2:end-1,3:end) - qy_new(2:end-1,1:end-2)) / (2*dy);
        dqx(1,2:end-1) = (qx_new(2,2:end-1) - qL_row(2:end-1)) / dx;
        dqy(1,2:end-1) = (qy_new(1,3:end) - qy_new(1,1:end-2)) / (2*dy);
        dqx(end,2:end-1) = (qR - qx_new(end-1,2:end-1)) / dx;
        dqy(end,2:end-1) = (qy_new(end,3:end) - qy_new(end,1:end-2)) / (2*dy);
        dqy(2:end-1,1) = (qy_new(2:end-1,2) - 0.0) / dy;
        dqx(2:end-1,1) = (qx_new(3:end,1) - qx_new(1:end-2,1)) / (2*dx);
        dqy(2:end-1,end) = (0.0 - qy_new(2:end-1,end-1)) / dy;
        dqx(2:end-1,end) = (qx_new(3:end,end) - qx_new(1:end-2,end)) / (2*dx);
        dqx(1,1) = (qx_new(2,1) - qL_row(1)) / dx;
        dqy(1,1) = (qy_new(1,2) - 0.0) / dy;
        dqx(1,end) = (qx_new(2,end) - qL_row(end)) / dx;
        dqy(1,end) = (0.0 - qy_new(1,end-1)) / dy;
        dqx(end,1) = (qR - qx_new(end-1,1)) / dx;
        dqy(end,1) = (qy_new(end,2) - 0.0) / dy;
        dqx(end,end) = (qR - qx_new(end-1,end)) / dx;
        dqy(end,end) = (0.0 - qy_new(end,end-1)) / dy;

        T_tm = T_tm - dt_sub * (dqx + dqy) ./ k_node;
        qx_tm = qx_new;
        qy_tm = qy_new;
    end
    T_time_tm(:,:,it) = T_tm;
end

figure(5); clf;
for k = 1:numel(snapshot_times)
    [~, idx] = min(abs(t_grid - snapshot_times(k)));
    subplot(1,numel(snapshot_times),k);
    imagesc(x_cv, y_cv, T_time_tm(:,:,idx)');
    axis equal tight;
    colorbar;
    xlabel('x^*');
    ylabel('y^*');
    title(sprintf('Thermomass T at t=%.3f', t_grid(idx)));
end
sgtitle('2D Thermomass temperature snapshots');
saveas(gcf, fullfile(outdir, 'minres_tm_snapshots.png'));

figure(6); clf;
imagesc(t_grid, x_cv, squeeze(T_time_tm(:,round((nely+1)/2),:)));
axis xy;
colorbar;
xlabel('t^*');
ylabel('x^*');
title('Thermomass spacetime at mid-y slice');
saveas(gcf, fullfile(outdir, 'minres_tm_spacetime.png'));

end