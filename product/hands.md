# 灵巧手 (Dexterous Hands)

本章节汇总了主流的机器人灵巧手，这是 VLA 算法落地的关键执行单元。

## Shadow Hand
- **厂商**: Shadow Robot (英国)
- **官网**: [https://www.shadowrobot.com/](https://www.shadowrobot.com/)
- **自由度**: 20 Actuated DOF (24 Total, 接近人手)。
- **特点**: **高端触觉研究首选**。极其灵活，集成 129 个传感器 (包括指尖触觉)，但价格昂贵，维护成本高。
- **参数**: 负载 5kg，自重 4.3kg，EtherCAT 通讯。
- **参考价格**: ~€110,000 (Full Kit)。

## Inspire RH56DFX
- **厂商**: Inspire Robots (因时机器人)
- **官网**: [http://en.inspire-robots.com/](http://en.inspire-robots.com/)
- **自由度**: 6 DOF (12 关节)。
- **特点**: **商业化成熟**。集成度高，支持力控，广泛用于科研和人形机器人集成。
- **参数**: 单指抓力 10-15N，重复精度 ±0.2mm，自重 ~600g。
- **参考价格**: ~$20,000 USD。

## Unitree Dex3-1
- **厂商**: Unitree (宇树科技)
- **官网**: [https://www.unitree.com/](https://www.unitree.com/)
- **自由度**: 7 DOF (3指设计)。
- **特点**: 专为 G1 机器人设计，力控驱动，适合通用抓取。
- **参数**: 负载 500g，自重 710g，USB 接口。
- **参考价格**: ~$5,200 - $6,500 USD。

## Agibot 灵犀 X1 (OmniHand)
- **厂商**: Agibot (智元机器人)
- **官网**: [https://www.agibot.com/](https://www.agibot.com/)
- **自由度**: 12 Active DOF (16 Total)。
- **特点**: **极致性价比**。刚柔耦合传动，适合低成本科研验证。
- **参数**: 负载 1kg，自重 ~500g，集成 400+ 触觉点 (Pro版)。
- **参考价格**: ~$2,000 USD (OmniHand 2025)。

## LEAP Hand
- **厂商**: CMU / LEAP 团队
- **官网**: [https://leap-hand.github.io/](https://leap-hand.github.io/)
- **自由度**: 16 DOF。
- **特点**: **开源项目**。成本低，Sim-to-Real 友好，学术界常用 (可自行 3D 打印)。
- **参数**: 组装成本 <$2,000，自重 ~600g。
- **参考价格**: DIY 成本 <$2,000 USD。

## Parsen DexH13
- **厂商**: Parsen (帕西尼感知)
- **官网**: [https://www.parsen.com.cn/](https://www.parsen.com.cn/)
- **自由度**: 13 Active + 3 Passive。
- **特点**: **触觉感知融合**。集成了数百个 ITPU 触觉单元 + 800万像素 AI 手眼相机，支持在手操作。

## Daimon DM-Hand1 (视触觉灵巧手)
- **厂商**: Daimon Robotics (戴盟机器人)
- **官网**: [http://www.daimonrobotics.com/](http://www.daimonrobotics.com/)
- **配置**: 仿人构型，指尖集成超薄视触觉传感器。
- **优势**: 解决了传统灵巧手触觉缺失或传感器过厚的问题，支持精细操作。

## Wonik Allegro Hand
- **厂商**: Wonik Robotics (韩国)
- **官网**: [https://www.wonikrobotics.com/](https://www.wonikrobotics.com/)
- **自由度**: 16 DOF (4指)。
- **特点**: 经典的科研用手，资料丰富，Sim-to-Real 验证多。
- **参数**: 负载 5kg，自重 1.8kg，CAN 总线。
- **参考价格**: ~€23,000。

---
[← Back to Product Index](./README.md)
