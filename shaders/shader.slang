//static float2 positions[3] = float2[](
//	float2(0.0, -0.5),
//	float2(0.5, 0.5),
//	float2(-0.5, 0.5)
//);

//static float3 colors[3] = float3[](
//	float3(1.0, 0.0, 0.0),
//	float3(0.0, 1.0, 0.0),
//	float3(1.0, 0.0, 1.0)
//);

struct VSInput {
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;

}

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
}
ConstantBuffer<UniformBuffer> ubo;

struct VSOutput {
    float4 sv_position : SV_Position;
    float3 fragColor;
    float2 fragTexCoord;
};


[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.sv_position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 1.0))));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    return output;
}

Sampler2D texture;
//: SV_Target means write this color to render target.
[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target
{
    return texture.Sample(vertIn.fragTexCoord);
}





