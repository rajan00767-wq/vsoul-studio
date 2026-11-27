# Studio Pose Verification Report (V13.3.1)

Audit of the V13.3 pose routing in `pipelines/uniform_swap/pose_analyzer.py`,
with emphasis on what the existing measurements can and cannot support.

---

## 1. Claims that do not survive audit

### 1.1 "Face frontalization"

V13.3's Route B applies a **2D similarity transform about the facial centre**.
That is an in-plane **roll correction**: it rotates the image so the eye line
is level. It does not rotate the head in 3D, cannot reveal a cheek hidden by
yaw, and changes no part of the face's 3D orientation.

Describing it as frontalization is wrong, and the distinction matters: a
portrait with 18° of yaw still has 18° of yaw after Route B, while the report
reads as though the pose was corrected.

**Correct description**: *identity-safe in-plane head-roll correction*.

### 1.2 "Eye gaze estimation"

The `off_camera_gaze` gate measures the offset of the **nose tip** from the
**eye midpoint**. That is a head-orientation proxy, not gaze. A subject can
hold their head square to camera and look 25° to the side with no change in
that statistic; conversely, a subject tracking the lens with their eyes while
turning their head registers as "off-camera gaze" when their gaze is correct.

The V13.3 report lists this gate as passing with a baseline of 0.47° and an
injected defect of 25.0°. Those numbers are real, but they measure what the
nose-to-eye-midpoint statistic does, **not** eye gaze. The gate should not be
cited as evidence that gaze is verified.

**Correct description**: *head-orientation proxy; not an authoritative gaze
measurement.* True iris-based gaze estimation with confidence — handling
glasses, reflections, partially closed eyes and low-resolution sources — is not
implemented, and implementing it is a V13.4 item per the specification.

## 2. What is genuinely distinguished today

| Quantity | Measured? | Basis | Confidence |
| :--- | :--- | :--- | :--- |
| Head roll (in-plane) | **Yes** | Eye-line angle from InsightFace landmarks | High; this is what 2D landmarks measure well |
| Head yaw | Partial | Landmark asymmetry proxy | Moderate; degrades at low resolution |
| Head pitch | Partial | Landmark proxy | Moderate |
| Eye gaze | **No** | nose↔eye-midpoint proxy only | **Not authoritative — do not cite as gaze** |
| Shoulder tilt | Yes | Foreground shoulder line | Moderate |
| Body orientation | No | — | Not measured separately from shoulder tilt |
| Neck alignment | Yes | Neck centroid offset | Moderate |

## 3. Identity preservation is not studio-pose compliance

These are separate properties and V13.3's reporting conflates them. An ArcFace
similarity of 0.99 against the source proves the delivered face is the same
person; it says nothing about whether the pose meets a studio standard. In
fact, for an off-camera source portrait, a **high** similarity to that source is
evidence the off-camera pose was *preserved* — the opposite of studio
compliance.

A portrait must therefore not be accepted as studio-ready merely because it
matches its source. Where the requested pose cannot be corrected safely, the
correct outcome is `needs_review` with the specific reason, not delivery.

## 4. Changes made in V13.3.1

**None to the pose code.** The existing identity-safe roll correction is
preserved exactly as specified. The specification explicitly assigns full 3D
frontalization and advanced gaze correction to V13.4, and forbids introducing
risky generative face reconstruction in V13.3.1.

What changed is the **reporting**: this document corrects the two
mis-descriptions above so that no downstream reader treats roll correction as
frontalization, or the nose-offset statistic as gaze.

## 5. Recommended V13.4 work

1. Iris-based gaze from landmark iris centres within the eye aperture, with an
   explicit confidence score and automatic abstention on glasses glare,
   partially closed eyes and sources below a resolution floor.
2. Separate reporting of identity similarity and pose compliance, so a
   delivered portrait carries both independently.
3. 3D-landmark-based yaw/pitch with calibrated error bars, replacing the
   current proxies.
