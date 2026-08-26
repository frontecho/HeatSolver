function plot_result_NCC_interactive(csvFile)
% Interactive viewer for result_NCC.csv.
% Left: slice distribution driven by pick mode and Lon/Lat/Depth sliders.
% Right: rotatable low-resolution 3D view clipped by the same sliders.

defaultViewAz = -35;  % 25;
defaultViewEl = 20;

if nargin < 1 || isempty(csvFile)
    csvFile = 'result-test04.csv';
    % csvFile = 'FEM/test1/result_NCC.csv';
end

if ~isfile(csvFile)
    error('File not found: %s', csvFile);
end

T = readtable(csvFile);
requiredVars = {
    'Lon', 'Lat', 'Depth_km', 'Temperature_K', ...
    'Thermal_Conductivity_W_mK', 'Heat_Production_W_m3', 'Heat_Flux_mW_m2'};
missingVars = setdiff(requiredVars, T.Properties.VariableNames);
if ~isempty(missingVars)
    error('Missing required columns: %s', strjoin(missingVars, ', '));
end

T = sortrows(T, {'Depth_km', 'Lat', 'Lon'});

fieldNames = {
    'Temperature_K', ...
    'Thermal_Conductivity_W_mK', ...
    'Heat_Production_W_m3', ...
    'Heat_Flux_mW_m2'};
fieldLabels = {
    'Temperature (K)', ...
    'Thermal Conductivity (W/m/K)', ...
    'Heat Production (W/m^3)', ...
    'Heat Flux (mW/m^2)'};

uniqueLon = unique(T.Lon);
uniqueLat = unique(T.Lat);
uniqueDepth = unique(T.Depth_km);
nLon = numel(uniqueLon);
nLat = numel(uniqueLat);
nDepth = numel(uniqueDepth);

[depthGroups, depthVals] = findgroups(T.Depth_km);
depthCounts = splitapply(@numel, T.Depth_km, depthGroups);
fullDepthVals = sort(depthVals(depthCounts == max(depthCounts)));
if numel(fullDepthVals) >= 2
    depthStep = median(diff(fullDepthVals));
else
    depthStep = median(diff(uniqueDepth(1:min(numel(uniqueDepth), 10))));
end
if ~isfinite(depthStep) || depthStep <= 0
    depthStep = 2;
end
regularDepth = (min(uniqueDepth):depthStep:floor(max(uniqueDepth) / depthStep) * depthStep)';
depthLevels = unique([regularDepth; max(uniqueDepth)]);
nDepthLevel = numel(depthLevels);

lonStep = median(diff(uniqueLon));
latStep = median(diff(uniqueLat));
if isempty(lonStep) || ~isfinite(lonStep)
    lonStep = 0.25;
end
if isempty(latStep) || ~isfinite(latStep)
    latStep = 0.25;
end

sliceMatrices = cell(numel(fieldNames), 1);
for f = 1:numel(fieldNames)
    sliceMatrices{f} = nan(nLat, nLon, nDepth);
end

for d = 1:nDepth
    depthMask = abs(T.Depth_km - uniqueDepth(d)) < 1e-9;
    sliceT = T(depthMask, :);
    [isMemberLon, lonIdx] = ismember(sliceT.Lon, uniqueLon);
    [isMemberLat, latIdx] = ismember(sliceT.Lat, uniqueLat);
    valid = isMemberLon & isMemberLat;
    linIdx = sub2ind([nLat, nLon], latIdx(valid), lonIdx(valid));
    for f = 1:numel(fieldNames)
        layer = sliceMatrices{f}(:, :, d);
        values = sliceT.(fieldNames{f});
        layer(linIdx) = values(valid);
        sliceMatrices{f}(:, :, d) = layer;
    end
end

[groups, lonVals, latVals] = findgroups(T.Lon, T.Lat);
minDepth = splitapply(@min, T.Depth_km, groups);
surfaceRows = false(height(T), 1);
for k = 1:numel(minDepth)
    rowIdx = find(T.Lon == lonVals(k) & T.Lat == latVals(k) & abs(T.Depth_km - minDepth(k)) < 1e-9, 1, 'first');
    if ~isempty(rowIdx)
        surfaceRows(rowIdx) = true;
    end
end
surfaceTable = T(surfaceRows, :);

state.fieldIdx = 1;
state.selectedLon = surfaceTable.Lon(round(height(surfaceTable) / 2));
state.selectedLat = surfaceTable.Lat(round(height(surfaceTable) / 2));
state.sliceLon = min(uniqueLon);
state.sliceLat = min(uniqueLat);
state.sliceDepth = min(uniqueDepth);

fig = figure(...
    'Name', 'NCC Interactive Result Viewer', ...
    'NumberTitle', 'off', ...
    'Color', 'w', ...
    'Units', 'normalized', ...
    'Position', [0.04, 0.05, 0.92, 0.86]);

hRotate = rotate3d(fig);
hRotate.Enable = 'on';

ctrlPanel = uipanel(fig, ...
    'Units', 'normalized', ...
    'Position', [0.04, 0.925, 0.34, 0.05], ...
    'BackgroundColor', 'w', ...
    'BorderType', 'none');

uicontrol(ctrlPanel, 'Style', 'text', 'String', 'Field', ...
    'Units', 'normalized', 'Position', [0.01, 0.28, 0.08, 0.45], ...
    'BackgroundColor', 'w', 'HorizontalAlignment', 'left', 'FontWeight', 'bold');

fieldPopup = uicontrol(ctrlPanel, 'Style', 'popupmenu', 'String', fieldLabels, ...
    'Units', 'normalized', 'Position', [0.10, 0.22, 0.34, 0.56], ...
    'Callback', @refreshAll);

uicontrol(ctrlPanel, 'Style', 'text', 'String', 'Pick mode', ...
    'Units', 'normalized', 'Position', [0.49, 0.28, 0.12, 0.45], ...
    'BackgroundColor', 'w', 'HorizontalAlignment', 'left', 'FontWeight', 'bold');

modePopup = uicontrol(ctrlPanel, 'Style', 'popupmenu', 'String', {'Lon Slice', 'Lat Slice', 'Depth Slice'}, ...
    'Units', 'normalized', 'Position', [0.63, 0.22, 0.34, 0.56], ...
    'Callback', @refreshAll);
modePopup.Value = 3;

infoText = uicontrol(fig, 'Style', 'text', ...
    'String', 'Click the left slice to open the nearest-neighbor profile.', ...
    'Units', 'normalized', 'Position', [0.39, 0.925, 0.16, 0.05], ...
    'BackgroundColor', 'w', 'HorizontalAlignment', 'left');

clipPanel = uipanel(fig, ...
    'Title', 'Slice / Clipping Controls', ...
    'Units', 'normalized', ...
    'Position', [0.04, 0.72, 0.40, 0.18], ...
    'BackgroundColor', 'w');

uicontrol(clipPanel, 'Style', 'text', 'String', 'Lon', ...
    'Units', 'normalized', 'Position', [0.03, 0.68, 0.10, 0.16], ...
    'BackgroundColor', 'w', 'HorizontalAlignment', 'left');
clipLonValue = uicontrol(clipPanel, 'Style', 'text', 'String', '', ...
    'Units', 'normalized', 'Position', [0.86, 0.68, 0.10, 0.16], ...
    'BackgroundColor', 'w', 'HorizontalAlignment', 'right');
clipLonSlider = uicontrol(clipPanel, 'Style', 'slider', ...
    'Min', 1, 'Max', nLon, 'Value', 1, ...
    'SliderStep', [1 / max(nLon - 1, 1), min(5 / max(nLon - 1, 1), 1)], ...
    'Units', 'normalized', 'Position', [0.14, 0.70, 0.70, 0.12], ...
    'Callback', @refreshAll);

uicontrol(clipPanel, 'Style', 'text', 'String', 'Lat', ...
    'Units', 'normalized', 'Position', [0.03, 0.40, 0.10, 0.16], ...
    'BackgroundColor', 'w', 'HorizontalAlignment', 'left');
clipLatValue = uicontrol(clipPanel, 'Style', 'text', 'String', '', ...
    'Units', 'normalized', 'Position', [0.86, 0.40, 0.10, 0.16], ...
    'BackgroundColor', 'w', 'HorizontalAlignment', 'right');
clipLatSlider = uicontrol(clipPanel, 'Style', 'slider', ...
    'Min', 1, 'Max', nLat, 'Value', 1, ...
    'SliderStep', [1 / max(nLat - 1, 1), min(5 / max(nLat - 1, 1), 1)], ...
    'Units', 'normalized', 'Position', [0.14, 0.42, 0.70, 0.12], ...
    'Callback', @refreshAll);

uicontrol(clipPanel, 'Style', 'text', 'String', 'Depth', ...
    'Units', 'normalized', 'Position', [0.03, 0.12, 0.10, 0.16], ...
    'BackgroundColor', 'w', 'HorizontalAlignment', 'left');
clipDepthValue = uicontrol(clipPanel, 'Style', 'text', 'String', '', ...
    'Units', 'normalized', 'Position', [0.86, 0.12, 0.10, 0.16], ...
    'BackgroundColor', 'w', 'HorizontalAlignment', 'right');
clipDepthSlider = uicontrol(clipPanel, 'Style', 'slider', ...
    'Min', 1, 'Max', nDepthLevel, 'Value', 1, ...
    'SliderStep', [1 / max(nDepthLevel - 1, 1), 1 / max(nDepthLevel - 1, 1)], ...
    'Units', 'normalized', 'Position', [0.14, 0.14, 0.70, 0.12], ...
    'Callback', @refreshAll);

axSlice = axes(fig, 'Units', 'normalized', 'Position', [0.06, 0.17, 0.40, 0.58]);
ax3D = axes(fig, 'Units', 'normalized', 'Position', [0.54, 0.13, 0.40, 0.73]);
sliceColorbar = [];
viewColorbar = [];

setAllowAxesRotate(hRotate, axSlice, false);
setAllowAxesRotate(hRotate, ax3D, true);

refreshAll();
disp('Interactive NCC viewer ready. Use Pick Mode + Lon/Lat/Depth sliders to draw the left slice. Click the left slice to open a nearest-neighbor profile.');

    function refreshAll(~, ~)
        state.fieldIdx = fieldPopup.Value;
        lonIdx = max(1, min(nLon, round(clipLonSlider.Value)));
        latIdx = max(1, min(nLat, round(clipLatSlider.Value)));
        depthIdx = max(1, min(nDepthLevel, round(clipDepthSlider.Value)));
        clipLonSlider.Value = lonIdx;
        clipLatSlider.Value = latIdx;
        clipDepthSlider.Value = depthIdx;

        state.sliceLon = uniqueLon(lonIdx);
        state.sliceLat = uniqueLat(latIdx);
        state.sliceDepth = depthLevels(depthIdx);

        set(clipLonValue, 'String', sprintf('%.2f', state.sliceLon));
        set(clipLatValue, 'String', sprintf('%.2f', state.sliceLat));
        set(clipDepthValue, 'String', sprintf('%.2f', state.sliceDepth));

        renderSlice();
        render3D();
    end

    function renderSlice()
        fieldName = fieldNames{state.fieldIdx};
        fieldLabel = fieldLabels{state.fieldIdx};
        modeStr = modePopup.String{modePopup.Value};

        cla(axSlice);

        switch modeStr
            case 'Lon Slice'
                subT = T(T.Lon == state.sliceLon, :);
                F = scatteredInterpolant(subT.Lat, subT.Depth_km, subT.(fieldName), 'natural', 'nearest');
                latGrid = uniqueLat;
                depthGrid = linspace(min(subT.Depth_km), max(subT.Depth_km), 240);
                [XQ, DQ] = meshgrid(latGrid, depthGrid);
                VQ = F(XQ, DQ);
                maxDepthByLat = accumarray(findgroups(subT.Lat), subT.Depth_km, [], @max);
                for kk = 1:numel(latGrid)
                    VQ(DQ(:, kk) > maxDepthByLat(kk), kk) = NaN;
                end
                imagesc(axSlice, latGrid, depthGrid, VQ);
                set(axSlice, 'YDir', 'reverse');
                axis(axSlice, 'tight');
                axis(axSlice, 'normal');
                daspect(axSlice, 'auto');
                pbaspect(axSlice, [1.8 1 1]);
                xlabel(axSlice, 'Lat');
                ylabel(axSlice, 'Depth (km)');
                title(axSlice, sprintf('Lon Slice | Lon=%.2f, Lat=%.2f, Depth=%.2f km | %s', ...
                    state.sliceLon, state.sliceLat, state.sliceDepth, fieldLabel));
                hold(axSlice, 'on');
                plot(axSlice, state.selectedLat, interpProfileDepth(T, state.selectedLon, state.selectedLat), 'kp', 'MarkerSize', 13, 'MarkerFaceColor', 'y', 'LineWidth', 1.0);
            case 'Lat Slice'
                subT = T(T.Lat == state.sliceLat, :);
                F = scatteredInterpolant(subT.Lon, subT.Depth_km, subT.(fieldName), 'natural', 'nearest');
                lonGrid = uniqueLon;
                depthGrid = linspace(min(subT.Depth_km), max(subT.Depth_km), 240);
                [XQ, DQ] = meshgrid(lonGrid, depthGrid);
                VQ = F(XQ, DQ);
                maxDepthByLon = accumarray(findgroups(subT.Lon), subT.Depth_km, [], @max);
                for kk = 1:numel(lonGrid)
                    VQ(DQ(:, kk) > maxDepthByLon(kk), kk) = NaN;
                end
                imagesc(axSlice, lonGrid, depthGrid, VQ);
                set(axSlice, 'YDir', 'reverse');
                axis(axSlice, 'tight');
                axis(axSlice, 'normal');
                daspect(axSlice, 'auto');
                pbaspect(axSlice, [1.8 1 1]);
                xlabel(axSlice, 'Lon');
                ylabel(axSlice, 'Depth (km)');
                title(axSlice, sprintf('Lat Slice | Lon=%.2f, Lat=%.2f, Depth=%.2f km | %s', ...
                    state.sliceLon, state.sliceLat, state.sliceDepth, fieldLabel));
                hold(axSlice, 'on');
                plot(axSlice, state.selectedLon, interpProfileDepth(T, state.selectedLon, state.selectedLat), 'kp', 'MarkerSize', 13, 'MarkerFaceColor', 'y', 'LineWidth', 1.0);
            case 'Depth Slice'
                depthIdxNearest = findNearestDepthIndex(uniqueDepth, state.sliceDepth);
                sliceMat = sliceMatrices{state.fieldIdx}(:, :, depthIdxNearest);
                imagesc(axSlice, uniqueLon, uniqueLat, sliceMat);
                set(axSlice, 'YDir', 'normal');
                axis(axSlice, 'equal');
                axis(axSlice, 'tight');
                pbaspect(axSlice, 'auto');
                xlabel(axSlice, 'Lon');
                ylabel(axSlice, 'Lat');
                title(axSlice, sprintf('Depth Slice | Lon=%.2f, Lat=%.2f, Depth=%.2f km | %s', ...
                    state.sliceLon, state.sliceLat, state.sliceDepth, fieldLabel));
                hold(axSlice, 'on');
                plot(axSlice, state.selectedLon, state.selectedLat, 'kp', 'MarkerSize', 13, 'MarkerFaceColor', 'y', 'LineWidth', 1.0);
        end

        grid(axSlice, 'on');
        colormap(axSlice, turbo);
        if ~isempty(sliceColorbar) && isgraphics(sliceColorbar)
            delete(sliceColorbar);
        end
        sliceColorbar = colorbar(axSlice);
        set(axSlice, 'ButtonDownFcn', @pickFromAxes);
        hImg = findobj(axSlice, 'Type', 'Image');
        if ~isempty(hImg)
            set(hImg(1), 'ButtonDownFcn', @pickFromAxes, 'PickableParts', 'all');
        end
    end

    function render3D()
        fieldName = fieldNames{state.fieldIdx};
        fieldLabel = fieldLabels{state.fieldIdx};

        clipMask = T.Lon >= state.sliceLon & T.Lat >= state.sliceLat & T.Depth_km >= state.sliceDepth;
        clipIdx = find(clipMask);
        if isempty(clipIdx)
            clipIdx = 1:height(T);
        end

        cla(ax3D);
        scatterHandle = scatter3(ax3D, T.Lon(clipIdx), T.Lat(clipIdx), -T.Depth_km(clipIdx), 14, T.(fieldName)(clipIdx), 'filled', ...
            'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.85);
        set(scatterHandle, 'PickableParts', 'none', 'HitTest', 'off');
        hold(ax3D, 'on');
        profileHandle = plot3(ax3D, state.selectedLon, state.selectedLat, -interpProfileDepth(T, state.selectedLon, state.selectedLat), 'kp', ...
            'MarkerSize', 14, 'MarkerFaceColor', 'y', 'LineWidth', 1.0);
        set(profileHandle, 'PickableParts', 'none', 'HitTest', 'off');
        zMin = -max(uniqueDepth);
        zMax = 0;
        lonLine = plot3(ax3D, [state.sliceLon, state.sliceLon], [min(uniqueLat), max(uniqueLat)], [zMax, zMax], 'k--', 'LineWidth', 1.0);
        latLine = plot3(ax3D, [min(uniqueLon), max(uniqueLon)], [state.sliceLat, state.sliceLat], [zMax, zMax], 'k--', 'LineWidth', 1.0);
        zline = -state.sliceDepth;
        depthLine = plot3(ax3D, [min(uniqueLon), max(uniqueLon)], [min(uniqueLat), min(uniqueLat)], [zline, zline], 'k--', 'LineWidth', 1.0);
        guideLine = plot3(ax3D, [state.sliceLon, state.sliceLon], [state.sliceLat, state.sliceLat], [zMin, zMax], ':', 'Color', [0.25, 0.25, 0.25], 'LineWidth', 1.0);
        set([lonLine, latLine, depthLine, guideLine], 'PickableParts', 'none', 'HitTest', 'off');
        grid(ax3D, 'on');
        view(ax3D, defaultViewAz, defaultViewEl);
        xlabel(ax3D, 'Lon');
        ylabel(ax3D, 'Lat');
        zlabel(ax3D, '-Depth (km)');
        title(ax3D, sprintf('Rotatable 3D View: %s', fieldLabel));
        colormap(ax3D, turbo);
        if ~isempty(viewColorbar) && isgraphics(viewColorbar)
            delete(viewColorbar);
        end
        viewColorbar = colorbar(ax3D);
    end

    function pickFromAxes(src, ~)
        ax = ancestor(src, 'axes');
        if isempty(ax)
            ax = src;
        end
        cp = ax.CurrentPoint;
        xClick = cp(1, 1);
        yClick = cp(1, 2);
        if ~isfinite(xClick) || ~isfinite(yClick)
            return;
        end

        modeStr = modePopup.String{modePopup.Value};
        switch modeStr
            case 'Lon Slice'
                lonClick = state.sliceLon;
                latClick = xClick;
            case 'Lat Slice'
                lonClick = xClick;
                latClick = state.sliceLat;
            otherwise  % Depth Slice
                lonClick = xClick;
                latClick = yClick;
        end

        dist2 = (surfaceTable.Lon - lonClick).^2 + (surfaceTable.Lat - latClick).^2;
        [~, idxMin] = min(dist2);
        state.selectedLon = surfaceTable.Lon(idxMin);
        state.selectedLat = surfaceTable.Lat(idxMin);

        set(infoText, 'String', sprintf('Selected Lon %.3f, Lat %.3f -> Profile', state.selectedLon, state.selectedLat));

        renderSlice();
        render3D();
        plotProfile(T, state.selectedLon, state.selectedLat, fieldNames{state.fieldIdx}, fieldLabels{state.fieldIdx});
    end
end

function depth0 = interpProfileDepth(T, lonSel, latSel)
subT = T(T.Lon == lonSel & T.Lat == latSel, :);
if isempty(subT)
    depth0 = 0;
else
    depth0 = min(subT.Depth_km);
end
end

function idx = findNearestDepthIndex(depthValues, targetDepth)
[~, idx] = min(abs(depthValues - targetDepth));
end

function plotProfile(T, lonSel, latSel, fieldName, fieldLabel)
subT = T(T.Lon == lonSel & T.Lat == latSel, :);
subT = sortrows(subT, 'Depth_km');
if isempty(subT)
    warning('No profile found at Lon %.4f, Lat %.4f.', lonSel, latSel);
    return;
end

fig = figure('Name', sprintf('Profile Lon %.3f Lat %.3f', lonSel, latSel), ...
    'NumberTitle', 'off', 'Color', 'w', 'Units', 'normalized', 'Position', [0.18, 0.12, 0.52, 0.7]);
tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile;
plot(ax1, subT.(fieldName), subT.Depth_km, '-o', 'LineWidth', 1.6, 'MarkerSize', 4, 'Color', [0.05, 0.35, 0.75]);
set(ax1, 'YDir', 'reverse');
grid(ax1, 'on');
xlabel(ax1, fieldLabel);
ylabel(ax1, 'Depth (km)');
title(ax1, sprintf('Profile at Lon %.3f, Lat %.3f', lonSel, latSel));

ax2 = nexttile;
plot(ax2, subT.Heat_Flux_mW_m2, subT.Depth_km, '-o', 'LineWidth', 1.6, 'MarkerSize', 4, 'Color', [0.75, 0.2, 0.1]);
set(ax2, 'YDir', 'reverse');
grid(ax2, 'on');
xlabel(ax2, 'Heat Flux (mW/m^2)');
ylabel(ax2, 'Depth (km)');
title(ax2, 'Heat Flux Along Profile');
sgtitle(fig, 'Detailed Vertical Profile');
end
