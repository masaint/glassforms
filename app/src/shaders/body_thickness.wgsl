// Integrate every interior voxel. Each 8x8 tile accumulates through the
// depth of the volume and returns four sums and counts for CPU readback.
struct Row {
    depth: vec4f,
    count: vec4u,
}
@group(0) @binding(0) var distances: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> rows: array<Row>;
var<workgroup> summed: array<vec4f, 64>;
var<workgroup> counted: array<vec4u, 64>;

@compute @workgroup_size(8, 8)
fn measure(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
    let n = textureDimensions(distances).x;
    let x = group.x * 8u + lane % 8u;
    let y = group.y * 8u + lane / 8u;
    var depth = vec4f(0.0);
    var count = vec4u(0u);
    if x < n && y < n {
        for (var z = 0u; z < n; z++) {
            let d = textureLoad(distances, vec3u(x, y, z), 0);
            depth += select(vec4f(0.0), -d, d < vec4f(0.0));
            count += select(vec4u(0u), vec4u(1u), d < vec4f(0.0));
        }
    }
    summed[lane] = depth;
    counted[lane] = count;
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        workgroupBarrier();
        if lane < stride {
            summed[lane] += summed[lane + stride];
            counted[lane] += counted[lane + stride];
        }
    }
    if lane == 0u {
        rows[group.y * ((n + 7u) / 8u) + group.x] = Row(summed[0], counted[0]);
    }
}
