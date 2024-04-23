//
//  ContentView.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 4/16/24.
//

import Combine
import SwiftUI
import RealityKit

struct ContentView : View {
    @StateObject var teleoperation = Teleoperation()

    @State private var _translationScale = 1.5

    var body: some View {
        ZStack {
            VStack {
                ARViewContainer(teleoperation: teleoperation)
                    .edgesIgnoringSafeArea(.all)
                    .onTouchDown {
                        teleoperation.transmitting = true
                    }
                    .onTouchUp {
                        teleoperation.transmitting = false
                    }
                    .onHorizontalDrag { startPosition, currentPosition in
                        // Wherever we started swiping is our 0 position
                        let distanceToRightEdge = 1.0 - startPosition
                        let pctToRight = (currentPosition - startPosition) / distanceToRightEdge
                        teleoperation.gripper = pctToRight
                    }
            }

            GeometryReader { geometry in
                VStack {
                    Text("Translation Scale")
                    HStack {
                        Slider(
                            value: $_translationScale,
                            in: 1...2
                        )
                        .onChange(of: _translationScale, initial: true) {
                            teleoperation.translationScale = Float(_translationScale)
                        }
                        Text("\(String(format: "%1.1f", _translationScale))")
                    }
                    Spacer()
                }
                .frame(width: geometry.size.width / 2) // set slider width to half of display width
            }
        }
    }
}

struct ARViewContainer: UIViewRepresentable {
    @ObservedObject private var _teleoperation: Teleoperation

    init(teleoperation: Teleoperation) {
        _teleoperation = teleoperation
    }

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)

        // Create a cube model
        let mesh = MeshResource.generateBox(size: 0.1, cornerRadius: 0.005)
        let material = SimpleMaterial(color: .gray, roughness: 0.15, isMetallic: true)
        let model = ModelEntity(mesh: mesh, materials: [material])
        model.transform.translation.y = 0.05

        // Create horizontal plane anchor for the content
        let anchor = AnchorEntity(.plane(.horizontal, classification: .any, minimumBounds: SIMD2<Float>(0.2, 0.2)))
        anchor.children.append(model)

        // Add the horizontal plane anchor to the scene
        arView.scene.anchors.append(anchor)

        // Subscribe events
        _teleoperation.subscribeToEvents(from: arView)

        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
    }
}

#Preview {
    ContentView()
}
